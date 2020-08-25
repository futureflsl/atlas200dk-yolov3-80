import numpy as np
import copy
import time
from atlasutil.presenteragent.presenter_types import *
import cv2 as cv

class Yolov3Manager(object):
    def __init__(self):
        self.anchors_yolo = [[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)],
                    [(10, 13), (16, 30), (33, 23)]]
        self.labels=[]

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-1 * x))
        return s

    # 获取分数最高的类别,返回分数和索引
    def getMaxClassScore(self, class_scores):
        class_score = 0
        class_index = 0
        for i in range(len(class_scores)):
            if class_scores[i] > class_score:
                class_index = i + 1
                class_score = class_scores[i]
        return class_score, class_index

    def getBBox(self, feat, anchors, image_shape, confidence_threshold):
        box = []
        for i in range(len(anchors)):
            for cx in range(feat.shape[0]):
                for cy in range(feat.shape[1]):
                    cf = feat[cx][cy][4 + 85 * i]
                    cp = feat[cx][cy][5 + 85 * i:85 + 85 * i]
                    b_confidence = self.sigmoid(cf)
                    b_class_prob = self.sigmoid(cp)
                    b_scores = b_confidence * b_class_prob
                    b_class_score, b_class_index = self.getMaxClassScore(b_scores)
                    if b_class_score > confidence_threshold:
                        tx = feat[cx][cy][0 + 85 * i]
                        ty = feat[cx][cy][1 + 85 * i]
                        tw = feat[cx][cy][2 + 85 * i]
                        th = feat[cx][cy][3 + 85 * i]
                        bx = (self.sigmoid(tx) + cx) / feat.shape[0]
                        by = (self.sigmoid(ty) + cy) / feat.shape[1]
                        bw = anchors[i][0] * np.exp(tw) / image_shape[0]
                        bh = anchors[i][1] * np.exp(th) / image_shape[1]
                        box.append([bx, by, bw, bh, b_class_score, b_class_index])
                   

                  

                    
                    
        return box

    # 非极大值抑制阈值筛选得到bbox
    def donms(self, boxes, nms_threshold):
        if len(boxes)==0:
            return []
        b_x = boxes[:, 0]
        b_y = boxes[:, 1]
        b_w = boxes[:, 2]
        b_h = boxes[:, 3]
        scores = boxes[:, 4]
        areas = (b_w + 1) * (b_h + 1)
        order = scores.argsort()[::-1]
        keep = []  # 保留的结果框集合
        while order.size > 0:
            i = order[0]
            keep.append(i)  # 保留该类剩余box中得分最高的一个
            # 得到相交区域,左上及右下
            xx1 = np.maximum(b_x[i], b_x[order[1:]])
            yy1 = np.maximum(b_y[i], b_y[order[1:]])
            xx2 = np.minimum(b_x[i] + b_w[i], b_x[order[1:]] + b_w[order[1:]])
            yy2 = np.minimum(b_y[i] + b_h[i], b_y[order[1:]] + b_h[order[1:]])
            # 相交面积,不重叠时面积为0
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # 相并面积,面积1+面积2-相交面积
            union = areas[i] + areas[order[1:]] - inter
            # 计算IoU：交 /（面积1+面积2-交）
            IoU = inter / union
            # 保留IoU小于阈值的box
            inds = np.where(IoU <= nms_threshold)[0]
            order = order[inds + 1]  # 因为IoU数组的长度比order数组少一个,所以这里要将所有下标后移一位

        final_boxes = [boxes[i] for i in keep]
        return final_boxes

    def getBoxes(self, resultList, anchors, img_shape, confidence_threshold, nms_threshold):
        boxes = []
        for i in range(len(resultList)):
            shape = resultList[i].shape
            resultList[i] = resultList[i].reshape(shape[0], shape[3], shape[1], shape[2]).transpose(0, 3, 2, 1)
            feature_map = resultList[i][0]
            #start = time.time()
            box = self.getBBox(feature_map, anchors[i], img_shape, confidence_threshold)    
            #getbboxtime = time.time()-start
            #print("getBBox time:",getbboxtime)
            boxes.extend(box)
        start = time.time()
        Boxes = self.donms(np.array(boxes), nms_threshold)
        nmstime = time.time()-start
        print("nms time:",nmstime)
        return Boxes

    def post_process(self, resultList, confidence_threshold, nms_threshold, model_shape, img_shape):
        boxes = self.getBoxes(resultList, self.anchors_yolo, model_shape, confidence_threshold, nms_threshold)
        detection_result_list = []
        for box in boxes:
            if self.labels != []:
                attr = self.labels[int(box[5])]
            else:
                attr = ""
            confidence = round(box[4], 4)
            ltx = int((box[0] - box[2] / 2) * img_shape[1])
            lty = int((box[1] - box[3] / 2) * img_shape[0])
            rbx = int((box[0] + box[2] / 2) * img_shape[1])
            rby = int((box[1] + box[3] / 2) * img_shape[0])
            result_text = str(attr) + " " + str(confidence * 100) + "%"
            detection_result_list.append([attr, ltx, lty, rbx, rby, result_text])
        return detection_result_list
    def inference(self, resultList, confidence_threshold, nms_threshold, model_shape, img_shape):
        start = time.time()
        boxes = self.getBoxes(resultList, self.anchors_yolo, model_shape, confidence_threshold, nms_threshold)
        boxtime = time.time()-start
        print("get boxes time:", boxtime)
        detection_result_list = []
        for box in boxes:
            detection_item = ObjectDetectionResult()
            if self.labels != []:
                detection_item.attr = self.labels[int(box[5])]
            else:
                detection_item.attr = ""
            detection_item.confidence = round(box[4],4)
            detection_item.lt.x = int((box[0]-box[2]/2)*img_shape[1])
            detection_item.lt.y = int((box[1]-box[3]/2)*img_shape[0])
            detection_item.rb.x = int((box[0]+box[2]/2)*img_shape[1])
            detection_item.rb.y = int((box[1]+box[3]/2)*img_shape[0])
            detection_item.result_text = str(detection_item.attr) + " " + str(detection_item.confidence*100) + "%"
            detection_result_list.append(detection_item)
        return detection_result_list


