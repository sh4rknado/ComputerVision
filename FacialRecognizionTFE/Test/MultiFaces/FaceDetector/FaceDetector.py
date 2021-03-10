# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import time

import cv2
import dlib
import numpy as np
import tensorflow as tf
import keras_vggface
from PIL import Image
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot

from scipy.special import expit

from FaceDetector import tiny_face_model
from FaceDetector import util


class FaceDetector:
    def __init__(self):
        # OpenCV HAAR
        self._faceCascade = cv2.CascadeClassifier('Data/Model/haarcascade_frontalface_default.xml')

        # OpenCV DNN supports 2 networks.
        # 1. FP16 version of the original caffe implementation ( 5.4 MB )
        # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
        DNN = "TF"

        if DNN == "CAFFE":
            self._modelFile = "Data/Model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            self._configFile = "Data/Model/deploy.prototxt"
            self._net = cv2.dnn.readNetFromCaffe(self._configFile, self._modelFile)
        else:
            self._modelFile = "Data/Model/opencv_face_detector_uint8.pb"
            self._configFile = "Data/Model/opencv_face_detector.pbtxt"
            self._net = cv2.dnn.readNetFromTensorflow(self._modelFile, self._configFile)

        self._conf_threshold = 0.8

        # DLIB HoG
        self._hogFaceDetector = dlib.get_frontal_face_detector()

        # DLIB MMOD
        self._dnnFaceDetector = dlib.cnn_face_detection_model_v1('Data/Model/mmod_human_face_detector.dat')

        # TinyFace
        self._MAX_INPUT_DIM = 5000.0
        self._prob_thresh = float(0.5)
        self._nms_tresh = float(0.1)
        self._lw = int(3)
        self._model = tiny_face_model.Model('Data/Model/hr_res101.weight')

    # ============================== < TinyFace Helpers > =======================================

    def _overlay_bounding_boxes(self, raw_img, refined_bboxes, lw):

        # Overlay bounding boxes on an image with the color based on the confidence.
        for r in refined_bboxes:
            _score = expit(r[4])
            cm_idx = int(np.ceil(_score * 255))
            rect_color = [int(np.ceil(x * 255)) for x in util.cm_data[cm_idx]]  # parula
            _lw = lw

            if lw == 0:  # line width of each bounding box is adaptively determined.
                bw, bh = r[2] - r[0] + 1, r[3] - r[0] + 1
                _lw = 1 if min(bw, bh) <= 20 else max(2, min(3, min(bh / 20, bw / 20)))
                _lw = int(np.ceil(_lw * _score))
                del bw
                del bh

            _r = [int(x) for x in r[:4]]
            cv2.rectangle(raw_img, (_r[0], _r[1]), (_r[2], _r[3]), rect_color, _lw)

            # Cleanning RAM
            del _score
            del cm_idx
            del rect_color
            del _lw
            del _r

    def _detectTinyFace(self, frame):
        x = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3])

        # Create the tiny face model which weights are loaded from a pretrained model.
        score_final = self._model.tiny_face(x)

        # Load an average image and clusters(reference boxes of templates).
        with open("Data/Model/hr_res101.weight", "rb") as f:
            _, mat_params_dict = pickle.load(f)

        average_image = self._model.get_data_by_key("average_image")
        clusters = self._model.get_data_by_key("clusters")
        clusters_h = clusters[:, 3] - clusters[:, 1] + 1
        clusters_w = clusters[:, 2] - clusters[:, 0] + 1
        normal_idx = np.where(clusters[:, 4] == 1)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            raw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_img_f = raw_img.astype(np.float32)

            def _calc_scales():
                raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
                min_scale = min(np.floor(np.log2(np.max(clusters_w[normal_idx] / raw_w))),
                                np.floor(np.log2(np.max(clusters_h[normal_idx] / raw_h))))
                max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / self._MAX_INPUT_DIM))
                scales_down = np.arange(min_scale, 0, 1.)
                scales_up = np.arange(0.5, max_scale, 0.5)
                scales_pow = np.hstack((scales_down, scales_up))
                scales = np.power(2.0, scales_pow)

                return scales

            scales = _calc_scales()
            start = time.time()

            # initialize output
            bboxes = np.empty(shape=(0, 5))

            # process input at different scales
            for s in scales:
                print("Processing at scale {:.4f}".format(s))
                img = cv2.resize(raw_img_f, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                img = img - average_image
                img = img[np.newaxis, :]

                # we don't run every template on every scale ids of templates to ignore
                tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
                ignoredTids = list(set(range(0, clusters.shape[0])) - set(tids))

                # run through the net
                score_final_tf = sess.run(score_final, feed_dict={x: img})

                # collect scores
                score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
                prob_cls_tf = expit(score_cls_tf)
                prob_cls_tf[0, :, :, ignoredTids] = 0.0

                def _calc_bounding_boxes():
                    # threshold for detection
                    _, fy, fx, fc = np.where(prob_cls_tf > self._prob_thresh)

                    # interpret heatmap into bounding boxes
                    cy = fy * 8 - 1
                    cx = fx * 8 - 1
                    ch = clusters[fc, 3] - clusters[fc, 1] + 1
                    cw = clusters[fc, 2] - clusters[fc, 0] + 1

                    # extract bounding box refinement
                    Nt = clusters.shape[0]
                    tx = score_reg_tf[0, :, :, 0:Nt]
                    ty = score_reg_tf[0, :, :, Nt:2 * Nt]
                    tw = score_reg_tf[0, :, :, 2 * Nt:3 * Nt]
                    th = score_reg_tf[0, :, :, 3 * Nt:4 * Nt]

                    # refine bounding boxes
                    dcx = cw * tx[fy, fx, fc]
                    dcy = ch * ty[fy, fx, fc]
                    rcx = cx + dcx
                    rcy = cy + dcy
                    rcw = cw * np.exp(tw[fy, fx, fc])
                    rch = ch * np.exp(th[fy, fx, fc])

                    scores = score_cls_tf[0, fy, fx, fc]
                    tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
                    tmp_bboxes = np.vstack((tmp_bboxes / s, scores))
                    tmp_bboxes = tmp_bboxes.transpose()

                    return tmp_bboxes

                tmp_bboxes = _calc_bounding_boxes()
                bboxes = np.vstack((bboxes, tmp_bboxes))  # <class 'tuple'>: (5265, 5)

            print("time {:.2f} secs ".format(time.time() - start))

            # refind_idx = util.nms(bboxes, nms_thresh)
            refind_idx = tf.image.non_max_suppression(tf.convert_to_tensor(value=bboxes[:, :4], dtype=tf.float32),
                                                      tf.convert_to_tensor(value=bboxes[:, 4], dtype=tf.float32),
                                                      max_output_size=bboxes.shape[0], iou_threshold=self._nms_tresh)
            refind_idx = sess.run(refind_idx)
            refined_bboxes = bboxes[refind_idx]

            if len(refined_bboxes) > 0:
                cv2.putText(raw_img, "TinyFaces : " + str(round(time.time() - start, 2)) + " s", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                # print("Face Detected : " + str(len(refind_idx)))
                self._overlay_bounding_boxes(raw_img, refined_bboxes, self._lw)

                # save image with bounding boxes
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            else:
                cv2.putText(raw_img, "TinyFaces : No face", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3, cv2.LINE_AA)

            return raw_img

    # ============================== < VGGFACE2 Helpers > =======================================

    # extract a single face from a given photograph
    def _extract_vggface(self, filename, required_size=(20, 20)):
        # load image from file
        pixels = pyplot.imread(filename)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        if len(results) > 0:
            # extract the bounding box from the first face
            x1, y1, width, height = results[0]['box']
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = np.asarray(image)
            return face_array
        else:
            return None

    # ============================== < FaceDetector  > =======================================

    def detectFaceOpenCVHaar(self, frame, inHeight=300, inWidth=0):
        frameOpenCVHaar = frame.copy()
        frameHeight = frameOpenCVHaar.shape[0]
        frameWidth = frameOpenCVHaar.shape[1]
        if not inWidth:
            inWidth = int((frameWidth / frameHeight) * inHeight)

        scaleHeight = frameHeight / inHeight
        scaleWidth = frameWidth / inWidth

        frameOpenCVHaarSmall = cv2.resize(frameOpenCVHaar, (inWidth, inHeight))
        frameGray = cv2.cvtColor(frameOpenCVHaarSmall, cv2.COLOR_BGR2GRAY)

        t1 = time.time()
        faces = self._faceCascade.detectMultiScale(frameGray)
        t2 = time.time()
        total_times = round(t2 - t1, 2)

        bboxes = []
        cv2.putText(frameOpenCVHaar, "OpenCV HaarCascade : " + str(total_times) + " s", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3, cv2.LINE_AA)

        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            cvRect = [int(x1 * scaleWidth), int(y1 * scaleHeight),
                      int(x2 * scaleWidth), int(y2 * scaleHeight)]
            bboxes.append(cvRect)
            cv2.rectangle(frameOpenCVHaar, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                          int(round(frameHeight / 150)), 4)
        return frameOpenCVHaar

    def detectFaceDlibHog(self, frame, inHeight=300, inWidth=0):

        frameDlibHog = frame.copy()
        frameHeight = frameDlibHog.shape[0]
        frameWidth = frameDlibHog.shape[1]

        if not inWidth:
            inWidth = int((frameWidth / frameHeight) * inHeight)

        scaleHeight = frameHeight / inHeight
        scaleWidth = frameWidth / inWidth

        frameDlibHogSmall = cv2.resize(frameDlibHog, (inWidth, inHeight))
        frameDlibHogSmall = cv2.cvtColor(frameDlibHogSmall, cv2.COLOR_BGR2RGB)

        t1 = time.time()
        faceRects = self._hogFaceDetector(frameDlibHogSmall, 0)
        t2 = time.time()
        total_times = round(t2-t1, 2)

        # print(frameWidth, frameHeight, inWidth, inHeight)
        bboxes = []
        cv2.putText(frameDlibHog, "HoG : " + str(total_times) + " s", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3, cv2.LINE_AA)
        for faceRect in faceRects:
            cvRect = [int(faceRect.left() * scaleWidth), int(faceRect.top() * scaleHeight),
                      int(faceRect.right() * scaleWidth), int(faceRect.bottom() * scaleHeight)]
            bboxes.append(cvRect)
            cv2.rectangle(frameDlibHog, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                          int(round(frameHeight / 150)), 4)
        return frameDlibHog

    def detectFaceDlibMMOD(self, frameDlibMMOD, inHeight=300, inWidth=0):
        frameHeight = frameDlibMMOD.shape[0]
        frameWidth = frameDlibMMOD.shape[1]

        if not inWidth:
            inWidth = int((frameWidth/frameHeight) * inHeight)

        frameDlibMMODSmall = cv2.resize(frameDlibMMOD, (inWidth, inHeight))
        frameDlibMMODSmall = cv2.cvtColor(frameDlibMMODSmall, cv2.COLOR_BGR2RGB)

        t1 = time.time()
        faceRects = self._dnnFaceDetector(frameDlibMMODSmall, 0)
        total_times = round((time.time() - t1), 2)
        del t1

        # print(frameWidth, frameHeight, inWidth, inHeight)
        # bboxes = []

        cv2.putText(frameDlibMMOD, "MMOD : " + str(total_times) + " s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3, cv2.LINE_AA)

        for faceRect in faceRects:
            cvRect = [int(faceRect.rect.left() * (frameWidth/inWidth)),
                      int(faceRect.rect.top() * (frameHeight/inHeight)),
                      int(faceRect.rect.right() * (frameWidth/inWidth)),
                      int(faceRect.rect.bottom() * (frameHeight/inHeight))]

            # Faces in bboxes
            # bboxes.append(cvRect)
            cv2.rectangle(frameDlibMMOD, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),
                          int(round(frameHeight / 150)), 4)
        # cleanning ram
        del frameWidth
        del frameHeight
        del inWidth
        del inHeight
        del faceRects

        return frameDlibMMOD

    def detectFaceOpenCVDnn(self, frame):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

        t1 = time.time()
        self._net.setInput(blob)
        detections = self._net.forward()
        t2 = time.time()
        total_times = round(t2-t1, 2)

        bboxes = []
        cv2.putText(frameOpencvDnn, "DNN : " + str(total_times) + " s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 3, cv2.LINE_AA)

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self._conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
        return frameOpencvDnn

    def detectTinyFace(self, frame):
        with tf.Graph().as_default():
            temp = self._detectTinyFace(frame)
        return temp

    def detectVGGFace(self, imgPath):
        # load the photo and extract the face
        pixels = self._extract_vggface(imgPath)

        if pixels is not None:
            # plot the extracted face
            pyplot.imshow(pixels)
            # show the plot
            pyplot.show()
