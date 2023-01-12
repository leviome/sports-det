import torch
import cv2
import sys
sys.path.append('./detectors/yolo7')

from api import Yolo7


if __name__ == '__main__':
    detector = Yolo7(ckpt='./detectors/yolo7/checkpoints/yolov7-e6e.pt')
    img = cv2.imread('./imgs/horses.jpg')
    res = detector.detect(img)
    print(res)
