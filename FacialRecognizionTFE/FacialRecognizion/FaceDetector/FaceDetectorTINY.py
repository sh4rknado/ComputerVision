# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===========================================================================
#           Definition of Import
# ===========================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.special import expit
from FaceDetector import tiny_face_model
from FaceDetector import util
import pickle
import time
import cv2
import numpy as np
import tensorflow as tf


# ===========================================================================
#           Infos developer
# ===========================================================================
__author__ = "Jordan BERTIEAUX"
__copyright__ = "Copyright 2020, Facial Recognition"
__credits__ = ["Jordan BERTIEAUX"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Jordan BERTIEAUX"
__email__ = "jordan.bertieaux@std.heh.be"
__status__ = "Production"


# ===========================================================================
#         Definition of Class FaceDetectorHoG
# ===========================================================================
class FaceDetectorTiny:
    def __init__(self, prob_thresh, nms_thres, lw, model):
        self._MAX_INPUT_DIM = 5000.0
        self._prob_thresh = float(prob_thresh)
        self._nms_tresh = float(nms_thres)
        self._lw = int(lw)
        self._model_path = model
        self._model = tiny_face_model.Model(model)

    # *=========================*
    # |  Extract Faces Process  |
    # *=========================*
    def detectFaceTiny(self, frame):
        with tf.Graph().as_default():
            x = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3])

            # Create the tiny face model which weights are loaded from a pretrained model.
            score_final = self._model.tiny_face(x)

            # Load an average image and clusters(reference boxes of templates).
            with open(self._model_path, "rb") as f:
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

                # initialize output
                bboxes = np.empty(shape=(0, 5))

                start = time.time()
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
                                                          max_output_size=bboxes.shape[0],
                                                          iou_threshold=self._nms_tresh)
                refind_idx = sess.run(refind_idx)
                refined_bboxes = bboxes[refind_idx]

                faces = []
                if len(refined_bboxes) > 0:
                    faces = self._GetFaces(raw_img, refined_bboxes, self._lw)

                return [faces, refined_bboxes]

    def DetectFace_Name(self, frame, names):
        with tf.Graph().as_default():
            x = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3])

            # Create the tiny face model which weights are loaded from a pretrained model.
            score_final = self._model.tiny_face(x)

            # Load an average image and clusters(reference boxes of templates).
            with open(self._model_path, "rb") as f:
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

                # initialize output
                bboxes = np.empty(shape=(0, 5))

                start = time.time()
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
                                                          max_output_size=bboxes.shape[0],
                                                          iou_threshold=self._nms_tresh)
                refind_idx = sess.run(refind_idx)
                refined_bboxes = bboxes[refind_idx]

                data = None
                if len(refined_bboxes) > 0:
                    data = self._overlay_bounding_boxes_names(raw_img, refined_bboxes, self._lw, names)

                return data

    def ExtractFace(self, frame, savepath):
        with tf.Graph().as_default():
            x = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3])

            # Create the tiny face model which weights are loaded from a pretrained model.
            score_final = self._model.tiny_face(x)

            # Load an average image and clusters(reference boxes of templates).
            with open(self._model_path, "rb") as f:
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

                # initialize output
                bboxes = np.empty(shape=(0, 5))

                start = time.time()
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
                                                          max_output_size=bboxes.shape[0],
                                                          iou_threshold=self._nms_tresh)
                refind_idx = sess.run(refind_idx)
                refined_bboxes = bboxes[refind_idx]

                if len(refined_bboxes) > 0:
                    self._overlay_bounding_boxes(raw_img, refined_bboxes, self._lw, savepath)

                return len(refined_bboxes)

    # ============================== < TinyFace Helpers > =======================================

    def _GetFaces(self, raw_img, refined_bboxes, lw):
        cpt = 0
        faces = []
        for r in refined_bboxes:
            _score = expit(r[4])
            _lw = lw

            if lw == 0:  # line width of each bounding box is adaptively determined.
                bw, bh = r[2] - r[0] + 1, r[3] - r[0] + 1
                _lw = 1 if min(bw, bh) <= 20 else max(2, min(3, min(bh / 20, bw / 20)))
                _lw = int(np.ceil(_lw * _score))
                del bw
                del bh
            _r = [int(x) for x in r[:4]]
            try:
                faces.append(raw_img[_r[1]:_r[3], _r[0]:_r[2]])
                cpt += 1
            finally:
                pass

            # Cleanning RAM
            del _score
            del _lw
            del _r

        del cpt
        return faces

    def _overlay_bounding_boxes_names(self, raw_img, refined_bboxes, lw, names):
        temp_name = ""
        cpt = 0
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
            cv2.rectangle(raw_img, (_r[0], _r[1]), (_r[2], _r[3]), (255, 0, 0), _lw)
            cv2.putText(raw_img, names[cpt], (_r[0], _r[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            temp_name = names[cpt]
            cpt += 1

            # Cleanning RAM
            del _score
            del cm_idx
            del rect_color
            del _lw
            del _r

        return [raw_img, temp_name]

    def SetName(self, raw_img, refined_bboxes, names):
        data = None
        if len(refined_bboxes) > 0:
            data = self._overlay_bounding_boxes_names(raw_img, refined_bboxes, self._lw, names)

        return data

    def _overlay_bounding_boxes(self, raw_img, refined_bboxes, lw, path):
        cpt = 0
        for r in refined_bboxes:
            _score = expit(r[4])
            _lw = lw

            if lw == 0:  # line width of each bounding box is adaptively determined.
                bw, bh = r[2] - r[0] + 1, r[3] - r[0] + 1
                _lw = 1 if min(bw, bh) <= 20 else max(2, min(3, min(bh / 20, bw / 20)))
                _lw = int(np.ceil(_lw * _score))
                del bw
                del bh
            _r = [int(x) for x in r[:4]]
            try:
                cv2.imwrite(path + str(cpt) + ".jpg", raw_img[_r[1]:_r[3], _r[0]:_r[2]])
                cpt += 1
            except:
                print("[ERROR] Can't write : " + path + str(cpt) + ".jpg")
            finally:
                pass

            # Cleanning RAM
            del _score
            del _lw
            del _r

        del cpt

