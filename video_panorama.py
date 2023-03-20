import os

import numpy as np
import cv2
import sys

sys.path.append('./detectors/yolo7')
sys.path.append('./visualization/')
sys.path.append('./utils/')

from api import Yolo7
from bbox_plot import plot_boxes, plot_box_in_tile, plain_plot

data_root = '/home/liweiliao/Downloads/panorama/'
INPUT_FILE = '/home/liweiliao/Downloads/panorama/record1.ts'
OUTPUT_FILE = data_root + 'track.mp4'

if __name__ == '__main__':
    print(sorted(os.listdir('/home/liweiliao/Downloads/panorama/')))
    detector = Yolo7(ckpt='./detectors/yolo7/checkpoints/yolov7-e6e.pt')

    writer = cv2.VideoWriter(OUTPUT_FILE,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             30,  # fps
                             (3632, 1632))

    reader = cv2.VideoCapture(INPUT_FILE)
    # (1632, 3632, 3)

    more = True
    c = 0
    start = 5000
    while more:
        more, frame = reader.read()
        c += 1
        if c <= start:
            continue

        if c % 1000 == 0:
            break

        res, player, window = detector.random_patch_det(frame, cls_ids=[32], row=2, col=4,
                                                        buffer=True,
                                                        frame_id=c - 1)

        frame = plot_boxes(frame, res)
        frame = plot_boxes(frame, player)
        # frame = plain_plot(frame, window)
        writer.write(frame)

        # cv2.imshow('0', frame)
        # cv2.waitKey(0)

        if c % 10 == 0:
            print(c)

    reader.release()
    writer.release()
    # cv2.destroyAllWindows()
