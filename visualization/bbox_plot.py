import cv2
from random import random
from classes import COCO_classes
from tools import deci
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color_list = []
for _ in range(len(COCO_classes)):
    color_list.append((int(128 * random()), int(128 * random()), int(255 * random())))
color_list[32] = (0, 255, 0)
color_list[0] = (255, 0, 125)


def plot_boxes(img, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box[0:4]
        cls = COCO_classes[int(box[-1])]
        prob = box[-2]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = color_list[int(box[-1])]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        caption = '%s %s' % (cls, deci(prob, precision=2))
        cv2.putText(img, caption, (x1, y1 - 5), font, fontScale, color, 1, cv2.LINE_AA)
    return img


def draw_poly(img, points, color=(255, 0, 0)):
    num_edges = len(points)
    for i, p in enumerate(points):
        p1 = p
        p2 = points[0] if i == num_edges - 1 else points[i + 1]
        img = cv2.line(img, p1, p2, color, 3)
    return img


def plain_plot(img, box, color=(0, 0, 255)):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
    return img


def plot_box_in_tile(img, box, vid=0):
    if len(box) == 0:
        blank = np.zeros((216, 320, 3), dtype=np.uint8)
        view = 'view_%02d' % vid
        cv2.putText(blank, view, (8, 12), font, fontScale, (255, 255, 0), 1, cv2.LINE_AA)
        return blank
    box = box[0]
    x1, y1, x2, y2 = box[0:4]
    cls = COCO_classes[int(box[-1])]
    prob = box[-2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    color = color_list[int(box[-1])]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    caption = '%s %s' % (cls, deci(prob, precision=2))
    cv2.putText(img, caption, (x1, y1 - 5), font, fontScale, color, 1, cv2.LINE_AA)
    y1 = max(108, y1)
    y1 = min(1080 - 108, y1)
    x1 = max(160, x1)
    x1 = min(1920 - 160, x1)
    img = img[y1 - 108:y1 + 108, x1 - 160:x1 + 160, :]
    view = 'view_%02d' % vid
    cv2.putText(img, view, (8, 12), font, fontScale, (255, 255, 0), 1, cv2.LINE_AA)

    return img
