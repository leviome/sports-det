import os
import sys
import json
import cv2

sys.path.append('./detectors/yolo7')
sys.path.append('./visualization/')
sys.path.append('./utils/')

from api import Yolo7
from bbox_plot import plain_plot

data_root = '/home/liweiliao/Downloads/panorama/'
INPUT_FILE = '/home/liweiliao/Downloads/panorama/src_5min.mp4'
OUTPUT_FILE = data_root + 'demo_5min.mp4'
BALL_LOC = 'ball_loc.json'
font = cv2.FONT_HERSHEY_SIMPLEX


def viz():
    input_stream = cv2.VideoCapture(INPUT_FILE)
    writer = cv2.VideoWriter(OUTPUT_FILE,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             30,  # fps
                             (3632, 1632))
    locations = json.loads(open(BALL_LOC, 'r').read())
    m = True
    fid = 0
    while m:
        print(fid)
        m, frm = input_stream.read()
        cv2.putText(frm, str(fid), (100, 100), font, 3, (255, 255, 0), 2, cv2.LINE_AA)
        if len(locations[str(fid)]) == 4:
            frm = plain_plot(frm, locations[str(fid)])
        fid += 1
        writer.write(frm)

    input_stream.release()
    writer.release()


def _main():
    detector = Yolo7(ckpt='./detectors/yolo7/checkpoints/yolov7-e6e.pt')
    ball_dict = dict()
    if os.path.exists(BALL_LOC):
        ball_dict = json.loads(open(BALL_LOC, 'r').read())

    reader = cv2.VideoCapture(INPUT_FILE)
    # (1632, 3632, 3)

    more = True
    c = 0
    buffer = ball_dict['0']
    while more:
        more, frame = reader.read()
        c += 1
        frame_id = c - 1

        # ball_dict[str(frame_id)] = []

        # res, player, window = detector.random_patch_det(frame, cls_ids=[32], row=2, col=4,
        #                                                 buffer=True,
        #                                                 frame_id=frame_id)
        # if len(ball_dict[str(frame_id)]) != 0:
        #     buffer = ball_dict[str(frame_id)]
        #     continue
        res = detector.roi_det(frame, prebox=buffer, cls_ids=[32], frame_id=frame_id)

        if len(res) == 0:
            buffer = []
            pass
        else:
            x1, y1, x2, y2 = [int(a) for a in res[0][0:4]]
            # if len(ball_dict[str(frame_id)]) == 0:
            ball_dict[str(frame_id)] = [x1, y1, x2, y2]
            buffer = [x1, y1, x2, y2]
            print(frame_id, buffer)

        if c % 10 == 0:
            print(c)
            with open(BALL_LOC, 'w') as f:
                f.write(json.dumps(ball_dict))

    reader.release()


if __name__ == '__main__':
    # _main()
    viz()
