import torch
import cv2
from models.experimental import attempt_load
from utils.datasets import preprocess
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import TracedModel
from math import sqrt
from open_clip.clip_api import ClipDiscriminator
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def center_patch(center, h, w, edge=640):
    x, y = center
    x1 = x - edge // 2
    x2 = x + edge // 2
    y1 = y - edge // 2
    y2 = y + edge // 2
    if x < edge // 2:
        x1, x2 = 0, 640
    if y < edge // 2:
        y1, y2 = 0, 640
    if x > w - edge // 2:
        x1, x2 = w - edge // 2, w
    if y > h - edge // 2:
        y1, y2 = h - edge // 2, h
    return int(x1), int(y1), int(x2), int(y2)


def distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def deduplicate(res, d_th=10):
    if len(res) <= 1:
        return res
    base = [res[0]]
    for i, b in enumerate(res[1:]):
        p1 = b[0:2]
        is_add = True
        for pp in base:
            p = pp[0:2]
            if distance(p, p1) <= d_th:
                is_add = False
        if is_add:
            base.append(b)
    return base
    # upper = []
    # for a in base:
    #     upper.append(a.unsqueeze(0))
    # return torch.cat(upper)


def filter_ball_size(res, ball_size_th=10):
    if len(res) <= 1:
        return res
    new_res = []
    for b in res:
        x1, y1, x2, y2 = b[0:4]
        w = x2 - x1
        h = y2 - y1
        if 100 > w >= ball_size_th and 100 > h >= ball_size_th:
            new_res.append(b)
    return new_res


class Yolo7:
    def __init__(self, ckpt='./yolo7/checkpoints/yolov7-e6e.pt', device='cuda:2', trace=False):
        self.device = device
        self.img_size = 640
        self.model = attempt_load(ckpt, map_location=device)
        if trace:
            self.model = TracedModel(self.model, device, self.img_size)  # img size is 640
        self.model.half()
        self.preprocess = preprocess
        self.buffer = []

        self.tics = 30
        self.tic_all = []
        self.stations = [[1092, 454, 1110, 476], [1366, 331, 1380, 345],
                         [655, 608, 674, 623], [3270, 908, 3288, 922]]

        self.status = 0
        self.official_ball = [100, 500]
        self.windows = [[0, 200, 1800, 1632], [900, 200, 2700, 1632], [1800, 200, 3600, 1632]]

        self.discriminator = ClipDiscriminator(["a basketball", "a player", "shoes", "other"], device=device)

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

    def tic(self, res):
        if self.tics > 20:
            self.tics -= 1
            self.tic_all += res

    def station_judge(self, res):
        if len(res) == 0 or self.tics == -1:
            return
        if 20 >= self.tics > 0:
            self.tics -= 1
            return
        for t in self.tic_all:
            for r in res:
                p1 = t[0:2]
                p2 = r[0:2]
                if distance(p1, p2) < 20:
                    self.stations.append(t)
                    continue
        self.tics -= 1
        self.stations = deduplicate(self.stations)

        print(self.stations)

    def de_station(self, res):
        if len(self.stations) == 0:
            return res
        new_res = []
        for b in res:
            is_add = True
            for s in self.stations:
                p1 = b[0:2]
                p2 = s[0:2]
                if distance(p1, p2) < 50:
                    is_add = False
            if is_add:
                new_res.append(b)
        return new_res

    def check_box(self):
        ...

    def random_patch_det(self, img, cls_ids=None, row=2, col=2, overlap=20, frame_id=0,
                         buffer=False, buffer_topk=3):
        height, width, _ = img.shape
        res_global = []  # self.detect(img, cls_ids=cls_ids)
        # players = self.detect(img, cls_ids=[0])
        players = []
        det_list = [res_global] if len(res_global) != 0 else []

        if buffer and len(self.buffer) > 0:
            # print(self.buffer)
            for box in self.buffer:
                x1, y1, x2, y2 = center_patch(box[0:2], height, width)
                # print(x1, y1, x2, y2)
                patch = img[y1:y2, x1:x2, :]
                res = self.detect(patch, cls_ids=cls_ids, offset=[x1, y1])
                if len(res) != 0:
                    det_list.append(res)

        h_edge = height // row
        w_edge = width // col
        assert h_edge >= self.img_size and w_edge >= self.img_size
        # print(h_edge, w_edge)
        margin = overlap // 2

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

        out = torch.cat(det_list) if len(det_list) != 0 else []
        if len(out) != 0:
            cache = []
            _, indices = torch.sort(out, descending=True, dim=0)
            for i, idx in enumerate(indices[:, -2]):
                cache.append(out[idx].unsqueeze(0))
            out = torch.cat(cache)
            out = deduplicate(out)
            out = filter_ball_size(out)
            # self.tic(out)
            # deduplicate(self.tic_all)
            # self.station_judge(out)
            out = self.de_station(out)
            if len(out) > 0:
                self.official_ball = out[0][0:2]
            cx = self.official_ball[0]
            if cx <= 1600:
                window = self.windows[0]
            elif cx <= 2500:
                window = self.windows[1]
            else:
                window = self.windows[2]

            if buffer:
                self.buffer = out[0:buffer_topk]
        else:
            window = self.windows[0]

        return out, players, window


if __name__ == '__main__':
    yolo = Yolo7('checkpoints/yolov7-e6e.pt', trace=False)
    print(yolo.img_size)
