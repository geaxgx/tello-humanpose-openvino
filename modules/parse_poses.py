#!/usr/bin/env python
"""

"""

import numpy as np

from pose_extractor import extract_poses

def get_root_relative_poses(inference_results, upsample_ratio, threshold):
    heatmap, paf_map = inference_results

    found_poses = extract_poses(heatmap[0:-1], paf_map, upsample_ratio)
    # scale coordinates to features space
    found_poses[:, 0:-1:3] /= upsample_ratio
    found_poses[:, 1:-1:3] /= upsample_ratio

    poses_2d = []
    num_kpt = 18

    for pose_id in range(found_poses.shape[0]):
        if found_poses[pose_id, 5] == -1:  # skip pose if does not found neck
            continue
        pose_2d = np.ones(num_kpt * 3 + 1, dtype=np.float32) * -1  # +1 for pose confidence
        for kpt_id in range(num_kpt):
            if found_poses[pose_id, kpt_id * 3] != -1:
                x_2d, y_2d, conf = found_poses[pose_id, kpt_id * 3:(kpt_id + 1) * 3]
                pose_2d[kpt_id * 3] = x_2d  # just repacking
                pose_2d[kpt_id * 3 + 1] = y_2d
                pose_2d[kpt_id * 3 + 2] = conf
        pose_2d[-1] = found_poses[pose_id, -1] # Global confidence
        poses_2d.append(pose_2d)
    poses_2d = np.array(poses_2d)

    
    return poses_2d



def parse_poses(inference_results, input_scale, stride, upsample_ratio, threshold=0.1):

    poses_2d = get_root_relative_poses(inference_results, upsample_ratio, threshold)
    poses_2d_scaled = []
    for pose_2d in poses_2d:
        num_kpt = (pose_2d.shape[0] - 1) // 3
        pose_2d_scaled = np.ones(pose_2d.shape[0], dtype=np.float32) * -1
        for kpt_id in range(num_kpt):
            if pose_2d[kpt_id * 3] != -1:
                pose_2d_scaled[kpt_id * 3] = pose_2d[kpt_id * 3] * stride / input_scale
                pose_2d_scaled[kpt_id * 3 + 1] = pose_2d[kpt_id * 3 + 1] * stride / input_scale
                pose_2d_scaled[kpt_id * 3 + 2] = pose_2d[kpt_id * 3 + 2]
        pose_2d_scaled[-1] = pose_2d[-1]
        poses_2d_scaled.append(pose_2d_scaled)

    return np.array(poses_2d_scaled)
