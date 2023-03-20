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
POLY = ((920, 480), (1800, 420), (2650, 550), (3350, 1080), (2700, 1600), (1100, 1600), (150, 950))
PLAYER_POLY = ((920, 480), (1800, 420), (2650, 550), (3350, 1080), (2700, 1600), (1100, 1600), (150, 950))


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


def filter_player_size(players, size_th=100):
    if len(players) <= 1:
        return players
    new_players = []
    for p in players:
        x1, y1, x2, y2 = p[0:4]
        w = x2 - x1
        h = y2 - y1
        if w > size_th or h > size_th:
            new_players.append(p)
    return new_players


def is_in_poly(p, polygon=POLY):
    """
    :param p: [x, y]
    :param polygon: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(polygon):
        next_i = i + 1 if i + 1 < len(polygon) else 0
        x1, y1 = corner
        x2, y2 = polygon[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


def court_filter(boxes, court=POLY):
    if len(boxes) == 0:
        return boxes
    new_boxes = []
    for box in boxes:
        x, y = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        if is_in_poly((x, y), polygon=court):
            new_boxes.append(box)
    return new_boxes


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
        self.stations = [[651, 608, 676, 625], [1102, 453, 1117, 470], [3270, 908, 3288, 922]]

        self.status = 0
        self.official_ball = [100, 500]
        self.windows = [[0, 200, 1800, 1632], [900, 200, 2700, 1632], [1800, 200, 3600, 1632]]

        self.discriminator = ClipDiscriminator(["a ball", "a man", "a player", "sports shoes", "other"], device=device)

    def detect(self, img, cls_ids=None, offset=None):
        if offset is None:
            offset = [0, 0]
        raw_size = img.shape
        img = self.preprocess(img, img_size=self.img_size, device=self.device, half=True)
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres=0.05, iou_thres=0.45, classes=cls_ids, agnostic=False)
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
                if distance(p1, p2) < 10:
                    is_add = False
                    break
            if is_add:
                new_res.append(b)
        return new_res

    def check_box(self, boxes, image):
        if len(boxes) == 0:
            return boxes
        new_boxes = []
        for box in boxes:
            coordinates = box[0:4]
            x1, y1, x2, y2 = [int(a) for a in coordinates]
            patch = image[y1 - 10:y2 + 10, x1 - 10:x2 + 10, :]
            prob = self.discriminator.forward(patch)
            print(prob)
            if prob[0] > .6:
                new_boxes.append(box)
        return new_boxes

    def roi_det(self, img, prebox=[], cls_ids=None, overlap=20, frame_id=0, buffer=False, buffer_topk=3):
        if len(prebox) == 0:
            out, _, _ = self.random_patch_det(img, cls_ids=[32], row=2, col=4, frame_id=frame_id)
            return out
        height, width, _ = img.shape
        res_global = []  # self.detect(img, cls_ids=cls_ids)
        det_list = [res_global] if len(res_global) != 0 else []

        if len(prebox) > 0:
            x1, y1, x2, y2 = center_patch(prebox[0:2], height, width)
            patch = img[y1:y2, x1:x2, :]
            res = self.detect(patch, cls_ids=cls_ids, offset=[x1, y1])
            if len(res) != 0:
                det_list.append(res[0])
        if len(det_list) == 0:
            return []
        out = self.de_station(det_list)
        out = self.check_box(out, img)
        return out

    def player_det(self, img, row=1, col=2, overlap=20):
        height, width, _ = img.shape
        h_edge = height // row
        w_edge = width // col
        assert h_edge >= self.img_size and w_edge >= self.img_size
        margin = overlap // 2
        det_list = []

        for i in range(row):
            for j in range(col):
                x1 = max(0, (j * w_edge - margin))
                y1 = max((0, i * h_edge - margin))
                x2 = min(width, ((j + 1) * w_edge + margin))
                y2 = min(height, ((i + 1) * h_edge + margin))
                patch = img[y1:y2, x1:x2, :]
                res = self.detect(patch, cls_ids=[0], offset=[x1, y1])
                res = res.detach().cpu().tolist()
                if len(res) != 0:
                    det_list += res
        det_list = filter_player_size(det_list)
        return det_list

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
                cache.append(out[idx].detach().cpu().tolist())
            out = deduplicate(cache)
            out = court_filter(out)
            # out = self.de_station(out)
            out = self.check_box(out, img)
            # out = filter_ball_size(out)

        #     if len(out) > 0:
        #         self.official_ball = out[0][0:2]
        #     cx = self.official_ball[0]
        #     if cx <= 1600:
        #         window = self.windows[0]
        #     elif cx <= 2500:
        #         window = self.windows[1]
        #     else:
        #         window = self.windows[2]
        #
        #     if buffer:
        #         self.buffer = out[0:buffer_topk]
        # else:
        #     window = self.windows[0]

        return out, players, []


if __name__ == '__main__':
    yolo = Yolo7('checkpoints/yolov7-e6e.pt', trace=False)
    print(yolo.img_size)
