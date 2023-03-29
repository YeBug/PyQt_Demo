# -*coding=utf-8

import os
import threading
import time
from typing import Union
import onnxruntime
import numpy as np
import torch
import torch.nn as nn
import cv2
import numpy
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from imgaug.augmenters import Resize
from utils.lane import Lane
from utils.grpc_inference import LaneDetClient
from nms import nms


class YOLOv5(object):
    def __init__(self, **kargs) -> None:
        self.initConfig(**kargs)
        self.inferencer = LaneDetClient()

    def initConfig(self, input_width=640, input_height=360, lane_color=(255, 0, 0), offset=72):
        self.input_width = input_width
        self.input_height = input_height
        self.lane_color = lane_color
        self.offset = offset
        self.anchor_ys = torch.linspace(1, 0, steps=self.offset, dtype=torch.float32)
        self.img_w = 1920
        self.img_h = 1080
        self.y_samples =  [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330,
         340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550,
          560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
    
    # def initModel(self, model_path):
    #     print("##############INIT MODEL#################")
    #     self.net = onnxruntime.InferenceSession(model_path, providers=['TensorrtExecutionProvider'])
    #     model_inputs = self.net.get_inputs()
    #     self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    #     model_outputs = self.net.get_outputs()
    #     self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    #     print(self.input_names)
    #     print(self.output_names)

    def detect(self, img_path):
        # TODO: no lane detected, try without transformer
        print("#####"+img_path+"#####")
        output = self.__inference(img_path=img_path)
        prediction = self.__decode(output)
        lanes = self.__pred2lanes(prediction)
        img = self.__drawLines(img_path, lanes[0])
        return img

    def __proccessInput(self, img_path):
        transformations = iaa.Sequential([Resize({'height': 360, 'width': 640})])
        transform = iaa.Sequential([iaa.Sometimes(then_list=[], p=0), transformations])
        img_org = cv2.imread(img_path)
        # img_org = cv2.resize(img_org, (0, 0), fx=2/3, fy=2/3)
        img_h, img_w = img_org.shape[0], img_org.shape[1]
        self.img_w, self.img_h = img_w, img_h
        img = cv2.resize(img_org, (0, 0), fx=1280/img_w, fy=720/img_h)
        img = transform(image=img_org.copy())
        img = img / 255.
        img = img.transpose(2, 0, 1)
        input_tensor = img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def __inference(self, img_path):
        input = self.__proccessInput(img_path)
        # output, anchors = self.net.run(self.output_names, {self.input_names[0]: input})
        output, anchors = self.inferencer(input)
        # anchors = self.anchors
        out_list = self.__nms(output, anchors)
        return out_list

    def __nms(self, batch_proposals, anchors, nms_thres=45, nms_topk=8, conf_threshold=0.2):
        anchors = anchors.squeeze(0)
        softmax = nn.Softmax(dim=1)
        batch_proposals = torch.Tensor(batch_proposals).cuda()
        proposals_list = []
        anchors = torch.from_numpy(anchors)
        for proposals in batch_proposals:
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            scores = softmax(proposals[:, :2])[:, 1]
            above_threshold = scores > conf_threshold
            proposals = proposals[above_threshold]
            scores = scores[above_threshold]
            anchor_inds = anchor_inds[above_threshold]
            if proposals.shape[0] == 0:
                proposals_list.append((proposals[[]], anchors[[]], None))
                continue
            keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
            keep = keep[:num_to_keep]
            proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            proposals_list.append((proposals, anchors[keep], anchor_inds))

        return proposals_list

    def __decode(self, proposals_list):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals, _,  _ in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            pred = self.__proposals_to_pred(proposals)
            decoded.append(pred)

        return decoded

    def __proposals_to_pred(self, proposals):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for lane in proposals:
            lane_xs = lane[5:] / self.input_width
            start = int(round(lane[2].item() * (self.offset-1)))
            length = int(round(lane[4].item()))
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)
            # end = label_end
            # if the proposal does not start at the bottom of the image,
            # extend its proposal until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) &
                       (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def __pred2lanes(self, preds):
        all_lanes = []
        for pred in preds:
            ys = np.array(self.y_samples) / 720
            lanes = []
            for lane in pred:
                xs = lane(ys)
                invalid_mask = xs < 0
                lane = (xs * 1280).astype(int)
                lane[invalid_mask] = -2
                lanes.append(lane.tolist())
            all_lanes.append(lanes)
        return all_lanes

    def __drawLines(self, img_file_path, pre_line=None, gt_line=None, resize=True):
        img = plt.imread(img_file_path)
        if resize:
            img = cv2.resize(img, (0, 0), fx=1280/self.img_w, fy=720/self.img_h)
        if pre_line:
            for line in pre_line:
                cv2.polylines(img, np.int32([list(tups for tups in zip(line, self.y_samples) if tups[0] > 0 )]), isClosed=False, color=(0, 255, 0), thickness=2) 
        if gt_line:
            for line in gt_line:
                cv2.polylines(img, np.int32([list(tups for tups in zip(line, self.y_samples) if tups[0] > 0 )]), isClosed=False, color=(0, 255, 255), thickness=2) 
        img = cv2.resize(img, (0, 0), fx=self.img_w/1280, fy=self.img_h/720)
        return img


class DataLoader(object):
    """逐帧加载图像，返回RGB格式"""
    VIDEO_TYPE = ('asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv')
    IMAGE_TYPE = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm')
    URL_TYPE = ('rtsp://', 'rtmp://', 'http://', 'https://')

    def __init__(self, source: Union[int, str], frame_skip=-1, flip=None, rotate=None, **kwargs):
        """
        :param source: input source
        :param frame_skip: frame skip or not, <0: auto; =0: dont skip; >0: skip  # video only
        :param flip:
        """
        self.source = source
        self.flip = flip
        self.rotate = rotate
        self.is_wabcam = self.source.isnumeric()
        self.is_video = self.source.lower().endswith(DataLoader.VIDEO_TYPE)
        self.is_image = self.source.lower().endswith(DataLoader.IMAGE_TYPE)
        self.is_screen = self.source.startswith('screen')
        self.is_url = self.source.lower().startswith(DataLoader.URL_TYPE)
        assert self.is_wabcam or self.is_video or self.is_image or self.is_screen or self.is_url, \
            f'Invalid or unsupported file format: {self.source}'
        self.cap = cv2.VideoCapture(self.source)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.idx = 0
        self.frame_skip = 0

    def __next__(self):
        if self.is_wabcam:
            ret, img = self.cap.read()
            path = ''
        elif self.is_video or self.is_image or self.is_url:
            while self.idx <= self.frame_skip:
                ret = self.cap.grab()
                self.idx += 1
                if not ret:
                    raise StopIteration
            ret, img = self.cap.retrieve()
            self.idx = 0
            path = self.source

        if not ret or img is None:
            raise StopIteration

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, path

    def __iter__(self):
        return self

    def __del__(self):
        if 'cap' in self.__dict__:
            self.cap.release()
