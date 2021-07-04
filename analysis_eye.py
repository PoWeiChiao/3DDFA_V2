import argparse
import csv 
import math

import yaml

def get_3d_points(csv_path, pts):
    with open(csv_path, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=':')
        for row in rows:
            pts.append(row[0])

def distance(pt1, pt2):
    total = 0
    p1 = pt1.split(',')
    p2 = pt2.split(',')
    for i in range(3): 
        total += (float(p1[i]) - float(p2[i]))**2
    return total**0.5

def main(args):
    pts = []

    get_3d_points(args.csv, pts)

    deviation_max = 0
    total_deviation = 0
    mean_deviation = 0
    for i in range(len(pts) // 2):
        total_deviation += distance(pts[2 * i], pts[2 * i + 1])
    
    mean_deviation = total_deviation / (len(pts) / 2)

    deviation_max = 0
    deviation = 0
    for i in range(len(pts) // 2):
        d = abs(distance(pts[2 * i], pts[2 * i + 1]) - mean_deviation)
        deviation += d
        if d > deviation_max:
            deviation_max = d
    
    print(mean_deviation)
    print(deviation / (len(pts) / 2))
    print(deviation_max)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The analysis of 3DDFA_V2')
    parser.add_argument('-c', '--csv', type=str, default='test_data/results/videos/Alex_mb1_eye.csv')
    args = parser.parse_args()
    main(args)