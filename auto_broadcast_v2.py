# auto broadcast by player positions
import math
import os
import sys
import json
import cv2
from time import sleep, time
from tqdm import tqdm

sys.path.append('./detectors/yolo7')
sys.path.append('./visualization/')
sys.path.append('./utils/')

from api import Yolo7, distance, is_in_poly, POLY, court_filter
from bbox_plot import plain_plot, draw_poly

data_root = '/home/liweiliao/Downloads/panorama/'
# INPUT_FILE = '/home/liweiliao/Downloads/panorama/src_5min.mp4'
INPUT_FILE = '/home/liweiliao/Downloads/panorama/record3.ts'
OUTPUT_FILE = data_root + 'record3_demo.mp4'
BALL_LOC = 'ball_loc.json'
PLAYER_LOC = 'player_loc.json'
FUSION = 'record3.json'
font = cv2.FONT_HERSHEY_SIMPLEX
key_importance = 10


def compute_iou(box1, box2):
    rec1 = [box1[1], box1[0], box1[3], box1[2]]
    rec2 = [box2[1], box2[0], box2[3], box2[2]]

    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def read_viz():
    results = json.loads(open("all_fusion_record3.json", 'r').read())
    # fusion_all_results = dict()
    # base = -1
    # key_x = -1
    # w_base = 0.95
    #
    # for k, v in tqdm(results.items()):
    #     x_axis = []
    #     frame_result = dict()
    #     frame_result["players"] = []
    #     frame_result["ball"] = v["ball"]
    #     frame_result["player_region"] = v["player_region"]
    #     frame_result["ball_region"] = v["ball_region"]
    #     frame_result["key_player"] = [] if "key_player" not in v.keys() else v["key_player"]
    #     if len(frame_result["key_player"]) > 0:
    #         key = frame_result["key_player"]
    #         key_x = (key[0] + key[2]) // 2
    #     for p in v["players"]:
    #         x_axis.append(p[0])
    #         frame_result["players"].append(p[0:4])
    #     x_axis = sorted(x_axis)
    #     x_mean = sum(x_axis) / len(x_axis) if key_x == -1 else key_x
    #     base = x_mean if base < 0 else base
    #     base = int(w_base * base + (1 - w_base) * x_mean)
    #
    #     frame_result["view_base_w095"] = base
    #     fusion_all_results[k] = frame_result
    # with open('all_fusion_record3.json', 'w') as f:
    #     f.write(json.dumps(fusion_all_results))
    #
    # return

    input_stream = cv2.VideoCapture(INPUT_FILE)
    writer = cv2.VideoWriter('/home/liweiliao/Downloads/panorama/read_viz.mp4',
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             30,  # fps
                             (3632, 1632))

    fid = 0
    more = True
    while more:
        more, frm = input_stream.read()
        players = results[str(fid)]["players"]
        ball = results[str(fid)]["ball"]
        key = results[str(fid)]["key_player"]
        base = results[str(fid)]["view_base_w095"]
        for p in players:
            frm = plain_plot(frm, p[0:4], color=(125, 0, 88))

        if len(key) > 0:
            frm = plain_plot(frm, key[0:4], color=(125, 255, 88))
        if len(ball) != 0:
            frm = plain_plot(frm, ball[0:4], color=(0, 0, 255))

        frm = cv2.line(frm, (base, 0), (base, 1600), (255, 0, 0), 3)

        fid += 1
        writer.write(frm)
        if fid % 10 == 0:
            print(fid)

    input_stream.release()
    writer.release()


def viz():
    filtered_results = dict()
    input_stream = cv2.VideoCapture(INPUT_FILE)
    writer = cv2.VideoWriter(OUTPUT_FILE,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             30,  # fps
                             (3632, 1632))
    results = json.loads(open(FUSION, 'r').read())
    w_base = 0.95
    m = True
    fid = 0
    base = -1
    key_buffer = None
    while m:
        print(fid)
        m, frm = input_stream.read()
        cv2.putText(frm, str(fid), (100, 100), font, 3, (255, 255, 0), 2, cv2.LINE_AA)
        x_axis = []

        current_results = dict()
        players, balls, key = results[str(fid)]["player"], results[str(fid)]["ball"], results[str(fid)]["key_player"]
        current_results["players"] = players
        # print(players)
        frm = plain_plot(frm, (100, 500, 3400, 1550), color=(0, 122, 0))
        # frm = plain_plot(frm, (300, 500, 3100, 1400), color=(255, 255, 255))
        frm = draw_poly(frm, POLY, color=(0, 0, 125))
        current_results["ball_region"] = POLY
        current_results["player_region"] = (100, 500, 3400, 1550)
        current_results["ball"] = []
        if len(balls) > 0:
            frm = plain_plot(frm, balls[0][0:4], color=(0, 0, 255))
            current_results["ball"] = balls[0][0:4]

        key_x = -1
        for i, box in enumerate(players):
            x_axis.append(box[0])
            frm = plain_plot(frm, box[0:4], color=(125, 0, 88))
            if i == key:
                key_buffer = box[0:4]
                current_results["key_player"] = key_buffer
                frm = plain_plot(frm, box[0:4], color=(125, 255, 88))
                key_x = box[0]

        # tracking
        miss = True
        if key_x == -1 and key_buffer is not None:
            for box in players:
                if key == -1:
                    if compute_iou(box[0:4], key_buffer) > 0.1:
                        key_buffer = box[0:4]
                        current_results["key_player"] = key_buffer
                        frm = plain_plot(frm, box[0:4], color=(125, 255, 88))
                        key_x = box[0]
                        miss = False
                        break
        # if miss:
        #     key_buffer = None

        x_axis = sorted(x_axis)
        x_axis = x_axis[1:-1]

        if key_x >= 0:
            ls = [key_x] * key_importance
            x_axis += ls
        x_mean = sum(x_axis) / len(x_axis) if key_x == -1 else key_x
        base = x_mean if base < 0 else base
        base = int(w_base * base + (1 - w_base) * x_mean)
        frm = cv2.line(frm, (base, 0), (base, 1600), (255, 0, 0), 3)
        filtered_results[str(fid)] = current_results
        if fid % 2 == 0:
            with open('result_fusion_record3.json', 'w') as f:
                f.write(json.dumps(filtered_results))

        fid += 1
        writer.write(frm)

    input_stream.release()
    writer.release()


def fusion(balls, players):
    margin = None
    key = -1
    if len(balls) == 0:
        return key, players
    else:
        b = balls[0][0:4]
        bx, by = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
        for i, p in enumerate(players):
            if margin is None:
                margin = (p[2] - p[0]) // 2
            if p[0] - margin <= bx <= p[2] + margin and p[1] - margin <= by <= p[3] + margin:
                key = i
                break
        return key, players


def position_filter(players, region=(100, 3400, 500, 1600)):
    new_players = []
    for p in players:
        x1, y1, x2, y2 = p[0:4]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        # if 100 < cx < 3400 and 450 < cy < 1600:
        if region[0] < cx < region[1] and region[2] < cy < region[3]:
            new_players.append(p)
    return new_players


def _main():
    detector = Yolo7(ckpt='./detectors/yolo7/checkpoints/yolov7-e6e.pt')

    reader = cv2.VideoCapture(INPUT_FILE)
    # (1632, 3632, 3)

    more = True
    frame_id = -1
    results = dict()
    tic = time()
    while more:
        try:
            # if frame_id == 3598:
            #     break
            more, frame = reader.read()
            frame_id += 1

            players = detector.player_det(frame)
            # players = position_filter(players, region=(100, 3400, 500, 1600))
            players = court_filter(players)

            # ball det
            balls, _, window = detector.random_patch_det(frame, cls_ids=[32], row=2, col=4,
                                                         buffer=True,
                                                         frame_id=frame_id)
            print(balls)
            print(players)

            key, players = fusion(balls, players)
            frame_result = dict()
            frame_result["player"] = players
            frame_result["ball"] = balls
            frame_result["key_player"] = key
            results[str(frame_id)] = frame_result

            # frame = plain_plot(frame, (100, 450, 3400, 1600))
            # frame = cv2.resize(frame, (1800, 800))

            # viz
            # for box in players:
            #     frame = plain_plot(frame, box[0:4])

            # debug steps
            # cv2.imshow('hh', frame)
            # cv2.waitKey(0)
            # sleep(1000)

            if frame_id % 10 == 0:
                print(frame_id)
                # with open(PLAYER_LOC, 'w') as f:
                with open(FUSION, 'w') as f:
                    f.write(json.dumps(results))
        except:
            print('over index')
    with open(FUSION, 'w') as f:
        f.write(json.dumps(results))
    toc = time()
    print(toc - tic)

    reader.release()


if __name__ == '__main__':
    # _main()
    # viz()
    read_viz()
