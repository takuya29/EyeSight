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


def getRep(imgPath, size):
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

    start = time.time()
    # print "aligned:"
    # print alignedFace
    gray_image = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2GRAY)
    hist_eq = cv2.equalizeHist(gray_image)

    detector = cv2.FeatureDetector_create('SIFT')

    keypoints = []
    for x, y in landmarks[36:48]:
        keypoints.append(cv2.KeyPoint(x, y, size))

    descripter = cv2.DescriptorExtractor_create("SIFT")
    k1, d1 = descripter.compute(gray_image, keypoints)
    # eye_image = alignedFace[:32, :]

    # print "gray"
    # print hist_eq
    # cv2.imwrite (os.path.join("gray",imgPath), gray_image)
    # cv2.imwrite (os.path.join("hist",imgPath), hist_eq)
    # cv2.imwrite (os.path.join("eye",imgPath), eye_image)

    start = time.time()

    return list(d1.reshape(12 * 128))


for size in range(1, 20):
    for dname in ['train', 'test']:
        f = open(
            '../input/sift_data/' + dname + '_sift_data' + str(size) + '.csv',
            'w+b')
        g = open(
            '../input/sift_data/' + dname + '_sift_error' + str(size) + '.csv',
            'w+b')
        dataWriter = csv.writer(f)
        errorwriter = csv.writer(g)

        for i in range(2, 4):
            file_list = glob.glob("mydata/" + dname + "/" + str(i) + "/*")
            output_list = []

            for file in file_list:
                output = getRep(file, size)

                if output is None:
                    errorwriter.writerow([file])
                else:
                    output_list.append([file, i] + output)

            dataWriter.writerows(output_list)

        f.close()
        g.close()

# [Label]
# 2 : Looking at Camera
# 3 : Not Looking at Camera
