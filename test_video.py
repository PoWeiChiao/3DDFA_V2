import argparse
import csv
import cv2
import numpy as np
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.functions import cv_draw_landmark

def eye_recorder(csv_name, ver):
    with open(csv_name + '_eye.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow([ver[0][39], ver[1][39], ver[2][39]])
        writer.writerow([ver[0][42], ver[1][42], ver[2][42]])

def csv_writer(csv_name, _t, elapse):
    with open(csv_name + '.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow([_t['forward_pass'].average_time])
        writer.writerow([_t['misc'].average_time])
        writer.writerow([elapse])

def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    if args.framework == 'caffe':
        print('caffe')
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_CAFFE import TDDFA_CAFFE

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_CAFFE(**cfg)
    elif args.framework == 'onnx':
        print('onnx')
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    elif args.framework == 'pytorch':
        print('pytorch')
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    cap = cv2.VideoCapture(args.video)

    target_name = f'test_data/results/videos/{args.video.split("/")[-1][:-4]}'
    if args.config == 'configs/mb1_120x120.yml':
        target_name += '_mb1'
    else:
        target_name += '_mb05'
    target_name += '_' + args.framework

    dense_flag = args.opt in ('2d_dense', '3d')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(target_name + '.avi', fourcc, 20.0, (800,  600))
    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break
        img_draw = frame.copy()
        boxes, _t = face_boxes(frame)
        if len(boxes) > 0:
            param_lst, roi_box_lst, elapse = tddfa(frame, boxes)
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
            for ver in ver_lst:
                img_draw = cv_draw_landmark(frame, ver)
                eye_recorder(target_name, ver)
                
            print('Detection: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(1, 1, _t['forward_pass'].average_time, _t['misc'].average_time))
            print('inference: {:.2f}ms'.format(elapse))
            csv_writer(target_name, _t, elapse)
        out.write(img_draw)
        cv2.imshow('frame', img_draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(target_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The test of webcam of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--video', type=str)
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-p', '--framework', type=str, default='pytorch')

    args = parser.parse_args()
    main(args)