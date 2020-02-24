"""
 2D Human Pose estimation in OpenVino environment
"""

from argparse import ArgumentParser
import os, sys

import cv2
import numpy as np
import logging as log


from modules.inference_engine import InferenceEngine
from modules.input_reader import InputReader
from modules.draw import draw_poses
from modules.parse_poses import parse_poses
from modules.pose import *
from modules.FPS import FPS

DEFAULT_MODEL='models/human-pose-estimation-0001.xml'

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

def build_argparser():
    parser = ArgumentParser(description='Human pose estimation demo. '
                                            'Press esc to exit, spacebar to (un)pause video or process next image.',
                                add_help=False)
    # args = parser.add_argument_group('Options')
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    parser.add_argument('-m', '--model',
                        help='Path to an .xml file with a trained model (dfault=%(default)s)',
                        type=str, default=DEFAULT_MODEL)
    parser.add_argument('-i', '--input',
                        help='Required. Path to input image, images, video file or camera id.',
                        nargs='+', default='')
    parser.add_argument('-d', '--device',
                        help='Specify the target device to infer on: CPU, GPU, FPGA, HDDL or MYRIAD. '
                            'The demo will look for a suitable plugin for device specified '
                            '(default=%(default)s)',
                        type=str, default='CPU')
    parser.add_argument('-s','--height_size', help='Network input layer height size. The smaller the faster (default=%(default)i)', type=int, default=256)
    parser.add_argument('-l','--log_level', type=str, default="INFO", help="Log level (default=%(default)s)")
    parser.add_argument('-u', '--upsample_ratio', help='Ratio of upsampling applied on the network output before performing grouping. '
                            'If value too small, accuracy can drop. The bigger the slower (default=%(default)i)',
                            type=int, default=4)
    parser.add_argument('-c', '--color_palette', type=str, default="openpose",
                        help="Colormaps used to draw skeletons. Can be a color '(B,G,R)', openpose or left_right (default=%(default)s)")

    return parser 

class HumanPose:
    def __init__(self, model=DEFAULT_MODEL, device='CPU', model_input_height=256, upsample_ratio=4, color_palette="openpose"):
        self.stride = 8
        self.inference_engine = InferenceEngine(model, device, self.stride)
        self.model_input_height = model_input_height
        self.upsample_ratio=upsample_ratio
        self.color_palette = color_palette

    def eval(self, frame):
        
        input_scale = self.model_input_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        # Inference
        inference_result = self.inference_engine.infer(scaled_img)
        # Postprocessing (grouping)
        poses_2d = parse_poses(inference_result, input_scale, self.stride, self.upsample_ratio) #, threshold)
            
        return poses_2d

    def draw(self, frame, poses_2d, draw_fps=False):
        draw_poses(frame, poses_2d, color_palette=self.color_palette)

   

if __name__ == '__main__':
    
 
    args = build_argparser().parse_args()
    
    if args.input == '':
        raise ValueError('Please, provide input data with argument "-i"')
    frame_provider = InputReader(args.input)
    hp = HumanPose(model=args.model, 
                    device=args.device, 
                    model_input_height=args.height_size, 
                    upsample_ratio=args.upsample_ratio,
                    color_palette=args.color_palette)
    
    delay = 1
    mean_time = -1
    fps = FPS()

    for frame in frame_provider:
        fps.update()
        current_time = cv2.getTickCount()
        poses_2d = hp.eval(frame)

        hp.draw(frame, poses_2d)
        fps.display(frame)
        cv2.imshow('Human Pose Estimation', frame)

        key = cv2.waitKey(delay)
        if key == 27: # Esc
            break
        if key == 32: # Space (pause/unpause)
            delay = (delay + 1) % 2

