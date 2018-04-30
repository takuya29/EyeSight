import argparse
import csv
import glob
import os
import time

import cv2
import numpy as np
import openface

start = time.time()
np.set_printoptions(precision=3)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('--dlibFacePredictor', type=str,
                    help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir,
                                         "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))


def fild_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)


def getRep(imgPath):
    if args.verbose:
        print("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    # print "bb",bb
    if bb is None:
        return None
        # raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print(
            "  + Face detection took {} seconds.".format(time.time() - start))

    landmarks = align.findLandmarks(rgbImg, bb)
    landmarks_np = np.array(landmarks, dtype=np.float32)

    # print "landmarks_np:",landmarks_np

    max_pt = np.max(landmarks_np, axis=0)
    min_pt = np.min(landmarks_np, axis=0)

    norm_landmarks = (landmarks_np - min_pt) / (max_pt - min_pt)
    # print "normalization :",norm_landmarks.reshape(136)

    start = time.time()
    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    # print "aligned:"
    # print alignedFace
    gray_image = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2GRAY)
    eye_image = gray_image[:32, :]

    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print(
            "  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()

    return eye_image.reshape(96 * 32).tolist()


for dname in ['train', 'test']:
    f = open('../input/not_hist/' + dname + '_data.csv', 'w+b')
    g = open('../input/not_hist/' + dname + '_error.csv', 'w+b')
    dataWriter = csv.writer(f)
    errorwriter = csv.writer(g)

    for i in range(2, 4):
        file_list = glob.glob("../mydata/" + dname + "/" + str(i) + "/*")
        output_list = []

        for file in file_list:
            output = getRep(file)
            if output is None:
                errorwriter.writerow([file])
            else:
                output_list.append([file, i] + output)

        # print output_list

        dataWriter.writerows(output_list)

    f.close()
    g.close()

# [Label]
# 2 : Looking at Camera
# 3 : Not Looking at Camera
