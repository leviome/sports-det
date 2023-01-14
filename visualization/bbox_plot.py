import cv2
from random import random
from classes import COCO_classes
from tools import deci

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color_list = []
for i in range(len(COCO_classes)):
    color_list.append((int(128 * random()), int(128 * random()), int(255 * random())))
color_list[32] = (0, 255, 0)


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
