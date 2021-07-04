import argparse
import csv 
import math

import yaml

def get_3d_points(csv_path, pts):
    with open(csv_path, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=':')
        count = 0
        pt = []
        for row in rows:
            pt.append(float(row[0]))
            count += 1
            if count == 3:
                count = 0
                pts.append(pt.copy())
                pt.clear()

def distance(pt1, pt2):
    total = 0
    for i in range(3): 
        total += (pt1[i] - pt2[i])**2
    return total**0.5

def main(args):
    pts1 = []
    pts2 = []

    get_3d_points(args.csv1, pts1)
    get_3d_points(args.csv2, pts2)

    deviation_max = 0
    total_deviation = 0
    for i in range(68):
        if i == 0:
            deviation_max = distance(pts1[i], pts2[i])
            total_deviation += deviation_max
        else:
            dis = distance(pts1[i], pts2[i])
            total_deviation += dis
            if dis > deviation_max:
                deviation_max = dis
    
    print(total_deviation / 68)
    print(deviation_max)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The analysis of 3DDFA_V2')
    parser.add_argument('-c1', '--csv1', type=str, default='test_data/results/images/down_mb1.csv')
    parser.add_argument('-c2', '--csv2', type=str, default='test_data/results/images/down_mb1_onnx.csv')
    args = parser.parse_args()
    main(args)