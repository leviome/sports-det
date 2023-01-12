import cv2
import torch
import sys

from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.datasets import preprocess

class Yolo7:
    def __init__(self, ckpt='./yolo7/checkpoints/yolov7-e6e.pt', device='cuda:2', trace=True):
        self.device = device
        self.img_size = 640
        self.model = attempt_load(ckpt, map_location=device)
        if trace:
            self.model = TracedModel(self.model, device, self.img_size) # img size is 640
        self.model.half()
        self.preprocess = preprocess

    def detect(self, img, cls_ids=None):
        raw_size = img.shape
        img = self.preprocess(img, img_size=self.img_size, device=self.device, half=True)
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=cls_ids, agnostic=False)
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], raw_size).round()
        return det



if __name__ == '__main__':
    yolo = Yolo7('checkpoints/yolov7-e6e.pt', trace=False)

