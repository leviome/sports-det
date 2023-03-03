import os

import numpy as np
import cv2
import sys

sys.path.append('./detectors/yolo7')
sys.path.append('./visualization/')
sys.path.append('./utils/')

from api import Yolo7
from bbox_plot import plot_boxes, plot_box_in_tile


def grid_plot():
    ...


if __name__ == '__main__':
    print(sorted(os.listdir('/amax/liwei/cba/acmp_data/cba-qingdao/images/frame_0001/')))
    detector = Yolo7(ckpt='./detectors/yolo7/checkpoints/yolov7-e6e.pt')
    tiles = []
    for i in range(31):
        img_name = '/amax/liwei/cba/acmp_data/cba-qingdao/images/frame_0017/view_%02d.jpg' % i
        img = cv2.imread(img_name)
        # img = cv2.imread('/home/liweiliao/Pictures/d1.jpg')
        # img = img[300:800, 500:1000, :]
        print(img.shape)
        # res = detector.detect(img, cls_ids=None)
        # res = detector.patch_det(img, cls_ids=[32])
        res = detector.random_patch_det(img, cls_ids=[32])
        img1 = plot_box_in_tile(img, res, vid=i)
        # img1 = plot_boxes(img, res)
        # cv2.resize(img1, (480, 320))
        # cv2.imshow('view_%02d' % i, img1)
        # cv2.waitKey(0)

        # print(res)
        tiles.append(img1)

    # rows = []
    # row = []
    # for n, tile in enumerate(tiles):
    #     if n % 6 == 0:
    #         if len(row) != 0:
    #             rows.append(np.hstack(row))
    #         row = []
    #     row.append(tile)
    p1 = np.hstack(tiles[0:6])
    p2 = np.hstack(tiles[6:12])
    p3 = np.hstack(tiles[12:18])
    p4 = np.hstack(tiles[18:24])
    p5 = np.hstack(tiles[24:30])
    overview = np.vstack((p1, p2, p3, p4, p5))

    # rows = (np.hstack(tiles[0:6]), np.hstack(tiles[6:12]),
    #         np.hstack(tiles[12:18]), np.hstack(tiles[18:24]),
    #         np.hstack(tiles[24:30]))
    # overview = np.stack(rows, axis=1)
    cv2.imwrite('frame_0017.jpg', overview)
    cv2.imshow('overview', overview)
    cv2.waitKey(0)
