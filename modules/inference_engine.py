#!/usr/bin/env python
"""
Inference on human pose model
Can work with the following modells of the Openvino model zoo:
 - human-pose-estimation-0001
 - human-pose-estimation-3d-0001 (but just the 2D part is used)
"""

import os

import numpy as np
import logging as log

from openvino.inference_engine import IENetwork, IECore

log = log.getLogger(__name__)

class InferenceEngine:
    def __init__(self, model_xml, device, stride):
        self.device = device
        self.stride = stride

        model_bin = os.path.splitext(model_xml)[0] + '.bin'
        log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
        self.net = IENetwork(model=model_xml, weights=model_bin)

        log.info("Loading Inference Engine")
        self.ie = IECore()
        log.info("Device info:")
        versions = self.ie.get_versions(device)
        log.info("{}{}".format(" "*8, device))
        log.info("{}MKLDNNPlugin version ......... {}.{}".format(" "*8, versions[device].major, versions[device].minor))
        log.info("{}Build ........... {}".format(" "*8, versions[device].build_number))

        self.input_blob = next(iter(self.net.inputs))
        log.info(f"Input blob: {self.input_blob} - shape: {self.net.inputs[self.input_blob].shape}")
        for o in self.net.outputs.keys():
            log.info(f"Output blob: {o} - shape: {self.net.outputs[o].shape}")
            if o == "Mconv7_stage2_L2":
                self.heatmaps_blob = "Mconv7_stage2_L2"
                self.pafs_blob = "Mconv7_stage2_L1"
            elif o == "heatmaps":
                self.heatmaps_blob = "heatmaps"
                self.pafs_blob = "pafs"
        log.info(f"Heatmaps blob: {self.heatmaps_blob} - PAFs blob: {self.pafs_blob}")

        log.info("Loading model to the plugin")
        self.exec_net = self.ie.load_network(network=self.net, num_requests=1, device_name=device)

    def infer(self, img):
        img = img[0:img.shape[0] - (img.shape[0] % self.stride), 0:img.shape[1] - (img.shape[1] % self.stride)]

        n, c, h, w = self.net.inputs[self.input_blob].shape
        
        if h != img.shape[0] or w != img.shape[1]:
            log.info(f"Reshaping of network")
            self.net.reshape({self.input_blob: (n, c, img.shape[0], img.shape[1])})
            log.info(f"Input blob: {self.input_blob} - new shape: {self.net.inputs[self.input_blob].shape}")
            for o in self.net.outputs.keys():
                log.info(f"Output blob: {o} - new shape: {self.net.outputs[o].shape}")
            self.exec_net = self.ie.load_network(network=self.net, num_requests=1, device_name=self.device)
        img = np.transpose(img, (2, 0, 1))[None, ]

        inference_result = self.exec_net.infer(inputs={self.input_blob: img})

        inference_result = (inference_result[self.heatmaps_blob][0], inference_result[self.pafs_blob][0])
        return inference_result
