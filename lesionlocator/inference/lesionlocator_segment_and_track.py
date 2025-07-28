import itertools
import os
from typing import List, Union
import warnings
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir
import numpy as np
import torch
from tqdm import tqdm
from natsort import natsorted

import lesionlocator
from lesionlocator.inference.export_prediction import export_prediction_from_logits
from lesionlocator.inference.lesionlocator_segment import LesionLocatorSegmenter
from lesionlocator.utilities.find_class_by_name import recursive_find_python_class
from lesionlocator.utilities.helpers import dummy_context
from lesionlocator.utilities.label_handling.label_handling import determine_num_input_channels
from lesionlocator.utilities.plans_handling.plans_handler import PlansManager

class LesionLocatorSegTracker(object):
    def __init__(self,
                 model_folder: str,
                 folds: tuple,
                 device: torch.device = torch.device('cuda')):
        dataset_json = load_json(join(model_folder, 'dataset.json'))
        plans = load_json(join(model_folder, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(folds, str):
            folds = [folds]

        parameters = []
        for i, f in enumerate(folds):
            print(f'Loading fold {f}')
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_folder, f'fold_{f}', "checkpoint_final.pth"),
                                    map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(lesionlocator.__path__[0], "training", "LesionLocatorTrainer"),
                                                    trainer_name, 'lesionlocator.training.LesionLocatorTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in lesionlocator.training.LesionLocatorTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            configuration_manager.patch_size,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters

        network.load_state_dict(parameters[0])
        self.network = network

        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        self.tile_step_size = 0.5
        self.use_gaussian = True
        self.use_mirroring = True
        self.device = device


    def mirror_and_predict(self, x0, x1, prompt):
        prediction = self.network(x0, x1, prompt, is_inference=True)
        if self.use_mirroring:
            mirror_axes = [2, 3, 4]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x0, axes), torch.flip(x1, axes), torch.flip(prompt, axes), is_inference=True), axes)
            prediction /= (len(axes_combinations) + 1)
        prediction = prediction[0]
        return prediction
    

    def track_single_lesion(self, lesion_index, bl, fu, bl_path, fu_path, output_folder, prompts, properties, preprocessor):
        im_name = os.path.basename(fu_path).replace(self.dataset_json['file_ending'],"")
        output_filename_truncated = join(output_folder, im_name + f'_lesion_{lesion_index}')

        if isinstance(prompts, list):
            bl, prompt, _ = preprocessor.run_case([bl_path],
                                            [prompts[lesion_index - 1]],
                                            self.plans_manager,
                                            self.configuration_manager,
                                            self.dataset_json)
            prompt = prompt[:]
            bl = bl[:].astype(np.half)
            if np.sum(prompt) == 0:
                print(f"Skipping lesion {lesion_index} in {fu_path} because no instance found")
                return
            with warnings.catch_warnings():
                # ignore 'The given NumPy array is not writable' warning
                warnings.simplefilter("ignore")
                bl = torch.from_numpy(bl).unsqueeze_(0).to(self.device)
                prompt = torch.from_numpy(prompt).unsqueeze_(0).to(self.device).half()
        else:
            prompt = torch.from_numpy(prompts == lesion_index).unsqueeze_(0).to(self.device).half()

        with torch.autocast(self.device.type, dtype=torch.float16, enabled=True) if self.device.type == 'cuda' else dummy_context():
            prediction = None
            for params in self.list_of_parameters: # fold iteration
                self.network.load_state_dict(params)
                self.network = self.network.to(self.device)
                self.network.eval()

                if prediction is None:
                    prediction = self.mirror_and_predict(bl, fu, prompt).to('cpu')
                else:
                    prediction += self.mirror_and_predict(bl, fu, prompt).to('cpu')

            if len(self.list_of_parameters) > 1:
                prediction /= len(self.list_of_parameters)

        export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager, self.dataset_json,
                        output_filename_truncated, False)
        return output_filename_truncated


    def track(self,
              baseline_image: str,
              follow_up_images: Union[str, List[str]],
              output_folder: str,
              prompt: Union[str, List[str]]):
        maybe_mkdir_p(output_folder)
        preprocessor = self.configuration_manager.preprocessor_class(verbose=False)

        if isinstance(follow_up_images, str):
            images = [baseline_image, follow_up_images]
        else:
            images = [baseline_image] + follow_up_images

        # Sliding window of 2 images at a time
        for bl_path, fu_path in zip(images[:-1], images[1:]):
            print(f'\n === Predicting {os.path.basename(fu_path)} === ')
            fu, _, properties = preprocessor.run_case([fu_path],
                                                        None,
                                                        self.plans_manager,
                                                        self.configuration_manager,
                                                        self.dataset_json)
            with warnings.catch_warnings():
                # ignore 'The given NumPy array is not writable' warning
                warnings.simplefilter("ignore")
                fu = fu[:].astype(np.half)
                fu = torch.from_numpy(fu).unsqueeze_(0).to(self.device)
            
            # If prompt is a list, we expect one filepath per lesion to track otherwise we assume
            # one file with all lesion instances (e.g. baseline instance segmentation labels)
            if isinstance(prompt, list):
                prompts = natsorted(prompt)
                lesion_indices = range(1, len(prompts) + 1)
                bl = None
            else:
                bl, bl_seg, _ = preprocessor.run_case([bl_path],
                                                        [prompt],
                                                        self.plans_manager,
                                                        self.configuration_manager,
                                                        self.dataset_json)
                prompts = bl_seg[:]
                lesion_indices = np.unique(bl_seg)
                lesion_indices = lesion_indices[lesion_indices > 0].astype(np.int16)
                if len(lesion_indices) == 0:
                    print(f"Skipping {fu_path} because no lesion to track")
                    continue

                bl = bl[:].astype(np.half)
                with warnings.catch_warnings():
                    # ignore 'The given NumPy array is not writable' warning
                    warnings.simplefilter("ignore")
                    bl = torch.from_numpy(bl).unsqueeze_(0).to(self.device)

            # Here, we iterate over the lesions to track
            with torch.no_grad():
                print(f"Predicting Lesions with indices: {lesion_indices}")

                predicted_files = []
                for lesion_index in tqdm(lesion_indices):
                    filename = self.track_single_lesion(lesion_index, bl, fu, bl_path, fu_path, output_folder, prompts, properties, preprocessor)
                    predicted_files.append(filename + self.dataset_json['file_ending'])
            torch.cuda.empty_cache()
            prompt = predicted_files


def segment_and_track():
    import argparse
    parser = argparse.ArgumentParser(description='This function handels the LesionLocator lesion segmentation and '
                                     'tracking inference using either point/box prompts or directly the baseline '
                                     'segmentation labels as prompt.')
    parser.add_argument('-bl', type=str, required=True,
                        help='Baseline scan path of the patient. File ending should be .nii.gz or specify another file_ending '
                        'in the dataset.json file of the downloaded checkpoint.')
    parser.add_argument('-fu', type=str, required=True, nargs='+',
                        help='Follow-up scan(s) of the patient. Can be either a single file path or an ordered list of file '
                        'paths separated by a comma to trigger autoregressive tracking. File ending should be .nii.gz or '
                        'specify another file_ending in the dataset.json file of the downloaded checkpoint.')
    parser.add_argument('-p', type=str, required=True, nargs='+',
                        help='Path to baseline segmentation(s) or prompt json. Expects one or a list of 3D volumetic masks (e.g .nii.gz)'
                        ' or a json file. If previous masks (single baseline instance segmentation map or list of semantic masks) are used, '
                        ' they must be in the same shape as the input images.')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If the folder does not exist it will be created. Predicted segmentations'
                             'will have the same name as their source images with the lesion instance as suffix.')
    parser.add_argument('-t', type=str, required=False, choices=['point', 'box', 'prev_mask'], default='prev_mask',
                        help='Specify the type of prompt. Options are "point", "box" or "prev_mask". Prompting with'
                        'point/box will trigger to first segment the baseline lesions adn then track. If previous mask is '
                        'used the model will directly track the provided baseline labels. Default: prev_mask')
    parser.add_argument('-m', type=str, required=True,
                        help='Folder of the LesionLocator model called "LesionLocatorCheckpoint"')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with.")

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using LesionLocator:\n"
        "Rokuss, M., Kirchhoff, Y., Akbal, S., Kovacs, B., Roy, S., Ulrich, C., ... & Maier-Hein, K. (2025).\n"
        "LesionLocator: Zero-Shot Universal Tumor Segmentation and Tracking in 3D Whole-Body Imaging. "
        "CVPR.\n#######################################################################\n")

    args = parser.parse_args()

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    
    if args.t != 'prev_mask':
        print("Segmenting baseline lesions from prompt...")
        assert len(args.p) == 1, "If using point/box prompt, only one prompt file is allowed."
        initial_segmenter = LesionLocatorSegmenter(tile_step_size=0.5,
                                use_gaussian=True,
                                use_mirroring=True,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=False,
                                allow_tqdm=True,
                                verbose_preprocessing=False)
        optimized_ckpt = "bbox_optimized" if args.t == 'box' else "point_optimized"
        checkpoint_folder = join(args.m, 'LesionLocatorSeg', optimized_ckpt)
        initial_segmenter.initialize_from_trained_model_folder(checkpoint_folder, args.f, "checkpoint_final.pth")
        print("Initiating segmentation.")
        predicted_files = initial_segmenter.predict_from_files(args.bl, args.o, args.p[0], args.t,
                                                            overwrite=True, num_processes_preprocessing=1,
                                                                num_processes_segmentation_export=3)
        prompt = predicted_files
    else:
        assert not args.p[0].endswith('.json'), "Prompt file must be a 3D volumetric mask (e.g .nii.gz) if using previous masks as prompt."
        prompt = args.p[0] if len(args.p) == 1 else args.p

    print("Tracking lesions (this may be time consuming for large images)...")
    predictor = LesionLocatorSegTracker(model_folder=join(args.m, 'LesionLocatorTrack'),
                                        folds=args.f,
                                        device=device)
    predictor.track(baseline_image=args.bl, follow_up_images=args.fu, output_folder=args.o, prompt=prompt)