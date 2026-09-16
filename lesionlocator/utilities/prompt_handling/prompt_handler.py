import os
import cc3d
import numpy as np
import torch

from skimage.morphology import ball
from batchgenerators.utilities.file_and_folder_operations import load_json

def sparse_to_dense_prompt(prompt, prompt_type, array):
    # For box prompt, prompt is a list of 6 integers, [zmin, zmax, ymin, ymax, xmin, xmax]
    if prompt_type == "box":
        assert len(prompt) == 6, "Box prompt must have 6 elements, i.e. [zmin, zmax, ymin, ymax, xmin, xmax], got {}".format(prompt)
        bbox_coords = [int(i) for i in prompt]
        if bbox_coords[1] <= bbox_coords[0] or bbox_coords[3] < bbox_coords[2] or bbox_coords[5] < bbox_coords[4]:
            print(f" Invalid bbox found! Skipping. bbox_coords: {bbox_coords}")
            return
        bb = torch.zeros_like(array)
        bb[:, bbox_coords[0]:bbox_coords[1], bbox_coords[2]:bbox_coords[3], bbox_coords[4]:bbox_coords[5]] = 1
        return bb
    # For point prompt, prompt is a list of 3 integers, [x, y, z]
    # We will place a ball of radius 5 around the point
    elif prompt_type == "point":
        assert not any(np.isnan(prompt)), "Point contains NaNs, got {}".format(prompt)
        centroid = np.round(prompt).astype(int)
        b = ball(5, strict_radius=False)
        ball_mask = np.zeros_like(array[0])
        xmin_img = max(0, centroid[0]-5)
        xmax_img = min(ball_mask.shape[0], centroid[0]+6)
        ymin_img = max(0, centroid[1]-5)
        ymax_img = min(ball_mask.shape[1], centroid[1]+6)
        zmin_img = max(0, centroid[2]-5)
        zmax_img = min(ball_mask.shape[2], centroid[2]+6)
        xmin_ball = 5 - (centroid[0] - xmin_img)
        xmax_ball = 5 + (xmax_img - centroid[0])
        ymin_ball = 5 - (centroid[1] - ymin_img)
        ymax_ball = 5 + (ymax_img - centroid[1])
        zmin_ball = 5 - (centroid[2] - zmin_img)
        zmax_ball = 5 + (zmax_img - centroid[2])
        ball_mask[xmin_img:xmax_img, ymin_img:ymax_img, zmin_img:zmax_img] = b[xmin_ball:xmax_ball, ymin_ball:ymax_ball, zmin_ball:zmax_ball]
        ball_mask = torch.from_numpy(ball_mask).to(array.dtype).unsqueeze_(0)
        return ball_mask
    else:
        raise ValueError(f"Unknown prompt type {prompt_type}")


def get_prompt_from_inst_or_bin_seg(seg, prompt_type):
    if prompt_type == "box":
        return get_bboxes_from_inst_or_bin_seg(seg)
    elif prompt_type == "point":
        return get_centroids_from_inst_or_bin_seg(seg)
    else:
        raise ValueError(f"Unknown prompt type {prompt_type}")
    

def get_centroids_from_inst_or_bin_seg(seg):
    uniques = np.unique(seg)
    uniques = uniques[uniques > 0]
    if np.any(uniques > 0):
        centroids = []
        if len(uniques) == 1 and uniques[0] == 1:
            assert len(seg.shape) == 4, "This function only works for 3d segmentations"
            seg_instances = cc3d.connected_components((seg[0] > 0).astype(np.uint8))
            stats = cc3d.statistics(seg_instances, no_slice_conversion=True)
            for centroid in stats["centroids"][1:]:
                centroids.append([centroid[0], centroid[1], centroid[2]])
        else:
            seg[seg < 0] = 0
            stats = cc3d.statistics(seg[0].astype(np.uint8), no_slice_conversion=True)
            for centroid_idx, centroid in enumerate(stats["centroids"][1:]):
                if centroid_idx + 1 not in uniques:
                    centroids.append([])
                    continue
                centroids.append([centroid[0], centroid[1], centroid[2]])
        return centroids
    else:
        return []


def get_bboxes_from_inst_or_bin_seg(seg):
        uniques = np.unique(seg)
        uniques = uniques[uniques > 0]
        if np.any(uniques>0):
            bboxes = []
            if len(uniques) == 1 and uniques[0] == 1:
                assert len(seg.shape) == 4, "This function only works for 3d segmentations"
                seg_instances = cc3d.connected_components((seg[0] > 0).astype(np.uint8))
                stats = cc3d.statistics(seg_instances, no_slice_conversion=True)
                for box in stats["bounding_boxes"][1:]:
                    # dilate box as we do in the other case, taking edges into account
                    box[0] = max(0, box[0].astype(np.int16) - 1)
                    box[1] = min(seg.shape[1] - 1, box[1] + 1) + 1
                    box[2] = max(0, box[2].astype(np.int16) - 1)
                    box[3] = min(seg.shape[2] - 1, box[3] + 1) + 1
                    box[4] = max(0, box[4].astype(np.int16) - 1)
                    box[5] = min(seg.shape[3] - 1, box[5] + 1) + 1
                    bboxes.append([box[0], box[1], box[2], box[3], box[4], box[5]])
            else:
                seg[seg < 0] = 0
                stats = cc3d.statistics(seg[0].astype(np.uint8), no_slice_conversion=True)
                for bbox_idx, box in enumerate(stats["bounding_boxes"][1:]):
                    if bbox_idx +1 not in uniques:
                        bboxes.append([])
                        continue
                    # dilate box as we do in the other case, taking edges into account
                    box[0] = max(0, box[0].astype(np.int16) - 1)
                    box[1] = min(seg.shape[1] - 1, box[1] + 1) + 1
                    box[2] = max(0, box[2].astype(np.int16) - 1)
                    box[3] = min(seg.shape[2] - 1, box[3] + 1) + 1
                    box[4] = max(0, box[4].astype(np.int16) - 1)
                    box[5] = min(seg.shape[3] - 1, box[5] + 1) + 1
                    bboxes.append([box[0], box[1], box[2], box[3], box[4], box[5]])
                    # This is [zmin, zmax, ymin, ymax, xmin, xmax]
            return bboxes
        else:
            return []
        

def get_prompt_from_json(json, prompt_type, data_properties, patch_size):
    json = load_json(json)
    if prompt_type == "box":
        return get_bboxes_from_json(json, data_properties, patch_size)
    elif prompt_type == "point":
        return get_centroids_from_json(json, data_properties, patch_size)
    else:
        raise ValueError(f"Unknown prompt type {prompt_type}")
    

def get_centroids_from_json(json, data_properties, patch_size):
    centroids = []
    for id in range(1, max(map(int, json.keys())) +1):
        point = json.get(str(id), None)
        if point is None:
            centroids.append([])
            continue
        else:
            point = [float(i) for i in point["point"]]
            z, y, x = point
            bbox_used_for_cropping = data_properties["bbox_used_for_cropping"]
            shape_after_cropping_and_before_resampling = data_properties["shape_after_cropping_and_before_resampling"]
            # Adapt the centroid to cropped data
            x = max(0, x - bbox_used_for_cropping[2][0])
            x = min(shape_after_cropping_and_before_resampling[2], x)
            y = max(0, y - bbox_used_for_cropping[1][0])
            y = min(shape_after_cropping_and_before_resampling[1], y)
            z = max(0, z - bbox_used_for_cropping[0][0])
            z = min(shape_after_cropping_and_before_resampling[0], z)

            # Adjust for resampling
            shape = patch_size
            factor = [shape[i] / shape_after_cropping_and_before_resampling[i] for i in range(3)]
            x = np.clip(np.round(x * factor[2]), 0, shape[2] - 1).astype(np.uint16)
            y = np.clip(np.round(y * factor[1]), 0, shape[1] - 1).astype(np.uint16)
            z = np.clip(np.round(z * factor[0]), 0, shape[0] - 1).astype(np.uint16)

            centroids.append([z, y, x])
    return centroids
    

def get_bboxes_from_json(json, data_properties, patch_size):
    bboxes = []
    for id in range(1, max(map(int, json.keys())) +1):
        box = json.get(str(id), None)
        if box is None:
            bboxes.append([])
            continue
        else:
            box = [float(i) for i in box["box"]]
            zmin, zmax, ymin, ymax, xmin, xmax = box
            bbox_used_for_cropping = data_properties["bbox_used_for_cropping"]
            shape_after_cropping_and_before_resampling = data_properties["shape_after_cropping_and_before_resampling"]
            # Adapt the bbox to cropped data

            xmin = max(0, xmin - bbox_used_for_cropping[2][0])
            xmax = min(shape_after_cropping_and_before_resampling[2], xmax - bbox_used_for_cropping[2][0])
            ymin = max(0, ymin - bbox_used_for_cropping[1][0])
            ymax = min(shape_after_cropping_and_before_resampling[1], ymax - bbox_used_for_cropping[1][0])
            zmin = max(0, zmin - bbox_used_for_cropping[0][0])
            zmax = min(shape_after_cropping_and_before_resampling[0], zmax - bbox_used_for_cropping[0][0])

            # Adjust for resampling
            shape = patch_size
            factor = [shape[i] / shape_after_cropping_and_before_resampling[i] for i in range(3)]
            xmin = np.floor(xmin * factor[2]).astype(np.uint16)
            xmax = np.ceil(xmax * factor[2]).astype(np.uint16)
            ymin = np.floor(ymin * factor[1]).astype(np.uint16)
            ymax = np.ceil(ymax * factor[1]).astype(np.uint16)
            zmin = np.floor(zmin * factor[0]).astype(np.uint16)
            zmax = np.ceil(zmax * factor[0]).astype(np.uint16)

            # Expand by 1 if fitting
            xmin = max(0, xmin - 1)
            xmax = min(patch_size[2], xmax + 1)
            ymin = max(0, ymin - 1)
            ymax = min(patch_size[1], ymax + 1)
            zmin = max(0, zmin - 1)
            zmax = min(patch_size[0], zmax + 1)

            # bbox_mask = np.zeros_like(data)
            # bbox_mask[:, xmin:xmax, ymin:ymax, zmin:zmax] = 1
            # bboxes.append(bbox_mask)
            bboxes.append([zmin, zmax, ymin, ymax, xmin, xmax])
    return bboxes