#!/usr/bin/env python3
"""Task 0. Initialize Yolo"""
from tensorflow.keras.models import load_model


class Yolo:
    """
    Documentation for class YOLO usage
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = load_model(model_path)
        with open(classes_path) as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
