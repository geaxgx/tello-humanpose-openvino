#!/usr/bin/env python
"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import cv2
import numpy as np
import re

from modules.pose import *



colors_left_right = [(0,255,255), (0,255,255), (0,255,0),
                    (0,255,0), (0,255,0), (0,0,255),
                    (0,0,255), (0,0,255), (0,255,0),
                    (0,255,0), (0,255,0), (0,0,255),
                    (0,0,255), (0,0,255), (0,255,0),
                    (0,0,255), (0,255,0), (0,0,255)]

colors_openpose = [(255, 0, 0), (255, 85, 0), (255, 170, 0),
                    (255, 255, 0), (170, 255, 0), (85, 255, 0),
                    (0, 255, 0), (0, 255, 85), (0, 255, 170),
                    (0, 255, 255), (0, 170, 255), (0, 85, 255),
                    (0, 0, 255), (85, 0, 255), (170, 0, 255),
                    (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def draw_poses(img, poses_2d, color_palette="openpose"):
    """
    Draw all the detected poses 
    """
    if re.fullmatch(r'\(\d+,\d+,\d+\)', re.sub(r'\s+', '', color_palette)):
        colors = [eval(color_palette)] * 18
    elif color_palette == "left_right":
        colors = colors_left_right
    else:
        colors = colors_openpose
    
    import pdb
    for pose in poses_2d:
        pose = np.array(pose[0:-1]).reshape((-1, 3)).transpose()
        was_found = pose[2] > 0
        for edge in body_edges:
            if was_found[edge[0]] and was_found[edge[1]]:
                cv2.line(img, tuple(pose[0:2, edge[0]].astype(np.int32)), tuple(pose[0:2, edge[1]].astype(np.int32)),
                         colors[edge[1]], 4, cv2.LINE_AA)
        for kpt_id in range(pose.shape[1]):
            if pose[2, kpt_id] != -1:
                cv2.circle(img, tuple(pose[0:2, kpt_id].astype(np.int32)), 5, colors[kpt_id], -1, cv2.LINE_AA)
