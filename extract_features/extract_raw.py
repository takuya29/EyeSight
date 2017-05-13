#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import glob
import csv

start = time.time()

import argparse
import cv2
import itertools
import os

import numpy as np
np.set_printoptions(precision=3)

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('--imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
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
    #print "bb",bb
    if bb is None:
    	return None
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    landmarks = align.findLandmarks (rgbImg, bb)
    landmarks_np = np.array(landmarks, dtype=np.float32)
    
    #print "landmarks_np:",landmarks_np

    max_pt = np.max(landmarks_np, axis=0)
    min_pt = np.min(landmarks_np, axis=0)

    norm_landmarks = (landmarks_np - min_pt) / (max_pt - min_pt)
    #print "normalization :",norm_landmarks.reshape(136)
    
    start = time.time()
    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    #print "aligned:"
    #print alignedFace
    gray_image = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2GRAY)
    hist_eq = cv2.equalizeHist(gray_image)
    eye_image = hist_eq[:, :]
    #eye_image = alignedFace[:32, :]

    #print "gray"
    #print hist_eq
    #cv2.imwrite (os.path.join("gray",imgPath), gray_image)
    #cv2.imwrite (os.path.join("hist",imgPath), hist_eq)
    #cv2.imwrite (os.path.join("eye",imgPath), eye_image)

    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()

    return eye_image.reshape(96*96).tolist()

    """
    rep = net.forward(alignedFace)
    #rep = net.forward(eye_image)
    if args.verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        print("Representation:")
        print("-----\n")
    return rep
    """

# for (img1, img2) in itertools.combinations(args.imgs, 2):
#     d = getRep(img1) - getRep(img2)
#     print("Comparing {} with {}.".format(img1, img2))
#     print(
#         "  + Squared l2 distance between representations: {:0.3f}".format(np.dot(d, d)))

for dname in ['train','test']:
    f = open('raw_data/'+dname+'_full_data.csv', 'w+b')
    g = open('raw_data/'+dname+'_full_error.csv', 'w+b')
    dataWriter = csv.writer(f)
    errorwriter = csv.writer(g)

    for i in range(2,4):
    	file_list = glob.glob("mydata/"+dname+"/"+str(i)+"/*")
    	output_list = []

    	for file in file_list:
    		output = getRep(file)
    		if output is None:
    			errorwriter.writerow([file])			
    		else:
    			output_list.append([file,i]+output)


    	#print output_list

    	dataWriter.writerows(output_list)

    f.close()
    g.close()

#[Label]
#2 : Looking at Camera
#3 : Not Looking at Camera