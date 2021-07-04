import argparse
import csv 
import math

import yaml

def get_time(csv_path, t):
    with open(csv_path, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=':')
        for row in rows:
            t.append(float(row[0]))

def main(args):
    t = []

    get_time(args.csv, t)

    count = 0
    face = 0
    face_max = 0
    nms = 0
    nms_max = 0
    dfa = 0
    dfa_max = 0
    for i in range(len(t)):
        if count == 0:
            face += t[i] * 1000
            if t[i] * 1000 > face_max:
                face_max = t[i] * 1000
            count += 1
        elif count == 1:
            nms =+ t[i] * 1000
            if t[i] * 1000 > nms_max:
                nms_max = t[i] * 1000
            count += 1
        elif count == 2:
            dfa += t[i]
            if t[i] > dfa_max:
                dfa_max = t[i]
            count = 0

    print(face / len(t))
    print(face_max)
    # print(nms / len(t))
    # print(nms_max)
    print(dfa / len(t)) 
    print(dfa_max)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The analysis of 3DDFA_V2')
    parser.add_argument('-c', '--csv', type=str, default='test_data/inputs/videos/Alex_bad_light_mb1.csv')
    args = parser.parse_args()
    main(args)