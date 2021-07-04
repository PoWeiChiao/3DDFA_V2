import sys
import argparse
import csv
import cv2
import glob
import os
import yaml

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
#from utils.render_ctypes import render  # faster
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, crop_img, parse_roi_box_from_bbox
from utils.tddfa_util import str2bool

def record_facebox_image(img, boxes, name):
    for box in boxes:
        roi_box = parse_roi_box_from_bbox(box)
        img = crop_img(img, roi_box)
        img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(name, img)

def csv_writer(csv_name, pts):
    with open(csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(pts[0][0, :])):
            writer.writerow([pts[0][0, :][i]])
            writer.writerow([pts[0][1, :][i]])
            writer.writerow([pts[0][2, :][i]])

def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    if args.framework == 'caffe':
        print('caffe')
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_CAFFE import TDDFA_CAFFE

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_CAFFE(**cfg)
    elif args.framework == 'onnx':
        print('onnx')
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

    for file in glob.glob(os.path.join(args.img_fp, '*.png')):
        img = cv2.imread(file)
        boxes, _t = face_boxes(img)

        n = len(boxes)
        if n == 0:
            print(f'No face detected, exit')
            sys.exit(-1)
        print(f'Detect {n} faces')

        param_lst, roi_box_lst, elapse = tddfa(img, boxes)
        dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')

        target_name = f'test_data/results/images/{file.split("/")[-1][:-4]}'
        if args.config == 'configs/mb1_120x120.yml':
            target_name += '_mb1'
        else:
            target_name += '_mb05'
        target_name += '_' + args.framework
        # record_facebox_image(img.copy(), boxes, target_name + '_face.jpg')

        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        pts = draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=target_name)
        csv_writer(target_name + '.csv', pts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='test_data/inputs/images/')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
    parser.add_argument('-p', '--framework', type=str, default='pytorch')

    args = parser.parse_args()
    main(args)