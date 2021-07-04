import argparse
import cv2
import numpy as np
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.functions import cv_draw_landmark

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

    cap = cv2.VideoCapture(0)
    dense_flag = args.opt in ('2d_dense', '3d')

    while (True):
        _, frame = cap.read()
        img_draw = frame.copy()
        boxes, _t = face_boxes(frame)
        if len(boxes) > 0:
            param_lst, roi_box_lst, elapse = tddfa(frame, boxes)
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
            for ver in ver_lst:
                img_draw = cv_draw_landmark(frame, ver)
            print('Detection: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(1, 1, _t['forward_pass'].average_time, _t['misc'].average_time))
            print('inference: {:.2f}ms'.format(elapse))
           
        cv2.imshow('frame', img_draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The test of webcam of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-p', '--framework', type=str, default='pytorch')

    args = parser.parse_args()
    main(args)