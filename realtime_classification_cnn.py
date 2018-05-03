import time

start = time.time()

import argparse
import cv2
import os

import numpy as np

np.set_printoptions(precision=2)
import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join('/home/shimada/openface/models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def getRep(bgrImg):
    start = time.time()
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    # Get the largest face bounding box
    # bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box

    # Get all bounding boxes
    bb = align.getAllFaceBoundingBoxes(rgbImg)

    if bb is None:
        # raise Exception("Unable to find a face: {}".format(imgPath))
        return None
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()

    alignedFaces = []
    for box in bb:
        face = align.align(
            args.imgDim,
            rgbImg,
            box,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.equalizeHist(face)
        alignedFaces.append(face[:32, :].astype(np.float32) / 255.)

    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    else:
        return (np.array(alignedFaces, dtype=np.float32), bb)


def infer(img, args):
    from cnn_train import CNN
    from chainer.serializers import load_npz
    import chainer.functions as F

    model = CNN()
    load_npz('output/cnn_model.npz', model, strict=True)

    repsAndBBs = getRep(img)
    reps = repsAndBBs[0]
    bbs = repsAndBBs[1]
    persons = []
    confidences = []
    reps = reps.reshape(-1, 1, 32, 96)

    if len(reps) == 0:
        return (confidences, bbs)

    predictions = F.softmax(model(reps))
    confidences = predictions.array[:, 0]

    return (confidences, bbs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument(
        '--captureDevice',
        type=int,
        default=0,
        help='Capture device. 0 for latop webcam and 1 for usb webcam')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--classifierModel',
        type=str,
        default='/home/shimada/openface/models/openface/celeb-classifier.nn4.small2.v1.pkl',
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')

    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(
        args.networkModel,
        imgDim=args.imgDim,
        cuda=args.cuda)

    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(args.captureDevice)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)
    video_capture.set(5, args.fps)

    out = cv2.VideoWriter('output/output.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), args.fps,
                          (args.width, args.height))

    confidenceList = []
    while True:
        ret, frame = video_capture.read()
        confidences, bbs = infer(frame, args)
        # print (" C: " + str(confidences))
        try:
            # append with two floating point precision
            confidenceList.append('%.2f' % confidences[0])
        except:
            # If there is no face detected, confidences matrix will be empty.
            # We can simply ignore it.
            pass

        notations = []
        for i, c in enumerate(confidences):
            if c <= args.threshold:  # 0.5 is kept as threshold for known face.
                notations.append("looking")
            else:
                notations.append("non-looking")

        # Print the person name and conf value on the frame next to the person
        # Also print the bounding box
        for idx, person in enumerate(notations):
            cv2.rectangle(frame, (bbs[idx].left(), bbs[idx].top()),
                          (bbs[idx].right(), bbs[idx].bottom()),
                          (255 * (confidences[idx]), 0,
                           255 * (1 - confidences[idx])), 2)
            cv2.putText(frame, "{} @{:.2f}".format(person, confidences[idx]),
                        (bbs[idx].left(), bbs[idx].bottom() + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('', frame)
        out.write(frame)
        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
