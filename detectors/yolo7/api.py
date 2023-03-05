import torch
import cv2
from models.experimental import attempt_load
from utils.datasets import preprocess
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import TracedModel
from math import sqrt


class Yolo7:
    def __init__(self, ckpt='./yolo7/checkpoints/yolov7-e6e.pt', device='cuda:2', trace=True):
        self.device = device
        self.img_size = 640
        self.model = attempt_load(ckpt, map_location=device)
        if trace:
            self.model = TracedModel(self.model, device, self.img_size)  # img size is 640
        self.model.half()
        self.preprocess = preprocess

    def detect(self, img, cls_ids=None, offset=None):
        if offset is None:
            offset = [0, 0]
        raw_size = img.shape
        img = self.preprocess(img, img_size=self.img_size, device=self.device, half=True)
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45, classes=cls_ids, agnostic=False)
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], raw_size).round()
        if offset != [0, 0]:
            det[:, [0, 2]] += offset[0]
            det[:, [1, 3]] += offset[1]
        return det

    def random_patch_det(self, img, cls_ids=None, row=2, col=2, overlap=20):
        res_global = [] # self.detect(img, cls_ids=cls_ids)
        height, width, _ = img.shape
        h_edge = height // row
        w_edge = width // col
        assert h_edge >= self.img_size and w_edge >= self.img_size
        # print(h_edge, w_edge)
        margin = overlap // 2

        det_list = [res_global] if len(res_global) != 0 else []

        for i in range(row):
            for j in range(col):
                x1 = max(0, (j * w_edge - margin))
                y1 = max((0, i * h_edge - margin))
                x2 = min(width, ((j + 1) * w_edge + margin))
                y2 = min(height, ((i + 1) * h_edge + margin))
                patch = img[y1:y2, x1:x2, :]
                res = self.detect(patch, cls_ids=cls_ids, offset=[x1, y1])
                if len(res) != 0:
                    det_list.append(res)

        return torch.cat(det_list) if len(det_list) != 0 else []


if __name__ == '__main__':
    yolo = Yolo7('checkpoints/yolov7-e6e.pt', trace=False)
