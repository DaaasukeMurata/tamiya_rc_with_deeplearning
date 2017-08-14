#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
from param_server import ParamServer


class MathLine():

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.piangle = self.__get_piangle()

    def __get_piangle(self):
        # y = tan(θ) * x + b
        vy = self.y2 - self.y1
        vx = self.x2 - self.x1
        return math.atan2(vy, vx) / math.pi


class MathLines():

    def __init__(self):
        self.lines = []

    def append(self, line):
        self.lines.append(line)

    def get_y_min(self):
        y_min = 100000
        for line in self.lines:
            if (line.piangle >= 0) and (line.y1 <= y_min):
                y_min = line.y1
                x_val = line.x1
            elif (line.piangle < 0) and (line.y2 <= y_min):
                y_min = line.y2
                x_val = line.x2
        return y_min, x_val

    def get_y_max(self):
        y_max = 0
        for line in self.lines:
            if (line.piangle >= 0) and (line.y2 >= y_max):
                y_max = line.y2
                x_val = line.x2
            elif (line.piangle < 0) and (line.y1 >= y_max):
                y_max = line.y1
                x_val = line.x1
        return y_max, x_val

    def get_rough_x(self, y, threshold=10):
        sum_x = 0
        count = 0
        for line in self.lines:
            if abs(line.y1 - y) <= threshold:
                sum_x += line.x1
                count += 1
            elif abs(line.y2 - y) <= threshold:
                sum_x += line.x2
                count += 1
        return sum_x / count

    def get_num(self):
        return len(self.lines)


class ProcessingImage():

    def __init__(self, img):
        self.img = img

    # 現在grayでも3channel colorで返す。
    def get_img(self):
        if len(self.img.shape) < 3:     # iplimage.shape is [x,y,colorchannel]
            return cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        else:
            return self.img

    def get_grayimg(self):
        if len(self.img.shape) < 3:     # iplimage.shape is [x,y,colorchannel]
            return self.img
        else:
            return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def __to_gray(self):
        if len(self.img.shape) == 3:     # iplimage.shape is [x,y,colorchannel]
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def __to_color(self):
        if len(self.img.shape) < 3:     # iplimage.shape is [x,y,colorchannel]
            self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

    def __threshold(self):
        self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)

    def __blur(self):
        FILTER_SIZE = (ParamServer.get_value('blur.gau_filter_size'),
                       ParamServer.get_value('blur.gau_filter_size'))
        # bilateralFilterだと色の差も加味する？
        # self.img = cv2.bilateralFilter(self.img, 5, 75, 75)
        self.img = cv2.GaussianBlur(self.img, FILTER_SIZE, 0)

    def __color_filter(self):
        LOW_B = ParamServer.get_value('color.low_b')
        LOW_G = ParamServer.get_value('color.low_g')
        LOW_R = ParamServer.get_value('color.low_r')
        HIGH_B = ParamServer.get_value('color.high_b')
        HIGH_G = ParamServer.get_value('color.high_g')
        HIGH_R = ParamServer.get_value('color.high_r')

        lower = np.array([LOW_B, LOW_G, LOW_R])
        upper = np.array([HIGH_B, HIGH_G, HIGH_R])

        hsv_image = cv2.cvtColor(self.get_img(), cv2.COLOR_BGR2HSV)
        mask_image = cv2.inRange(hsv_image, lower, upper)
        self.img = cv2.bitwise_and(self.get_img(), self.get_img(), mask=mask_image)
        area = cv2.countNonZero(mask_image)
        return area

    def __detect_edge(self):
        if ParamServer.get_value('edge.canny'):
            EDGE_TH_LOW = ParamServer.get_value('edge.canny_th_low')
            EDGE_TH_HIGH = ParamServer.get_value('edge.canny_th_high')
            self.img = cv2.Canny(self.img, EDGE_TH_LOW, EDGE_TH_HIGH)

        if ParamServer.get_value('edge.findContours'):
            self.__to_gray()
            # contours, hierarchy = cv2.findContours(self.img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # contours, hierarchy = cv2.findContours(self.img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.__to_color()
            cv2.drawContours(self.img, contours, -1, (255, 255, 255), 4)

    def __mask(self, vertices):
        # defining a blank mask to start with
        mask = np.zeros_like(self.img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(self.img.shape) > 2:
            channel_count = self.img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        vertices[0][0:, 0] = vertices[0][0:, 0] * self.img.shape[1]
        vertices[0][0:, 1] = vertices[0][0:, 1] * self.img.shape[0]

        int_vertices = vertices.astype(np.int32)

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, int_vertices, ignore_mask_color)

        # trancerate the image only where mask pixels are nonzero
        self.img = cv2.bitwise_and(self.img, mask)

    def __houghline(self):
        THRESHOLD = ParamServer.get_value('houghline.threshold')
        MIN_LINE_LENGTH = ParamServer.get_value('houghline.min_line_length')
        MAX_LINE_GAP = ParamServer.get_value('houghline.max_line_gap')
        self.__to_gray()
        return cv2.HoughLinesP(self.img, 1, np.pi / 180, THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP)

    def __extrapolation_lines(self, lines):

        if lines is None:
            return None, None

        # 検出する線の傾き範囲
        EXPECT_FRONT_LINE_M_MIN = ParamServer.get_value('extrapolation_lines.front_m_min')
        EXPECT_FRONT_LINE_M_MAX = ParamServer.get_value('extrapolation_lines.front_m_max')
        EXPECT_LEFT_LINE_M_MIN = ParamServer.get_value('extrapolation_lines.left_m_min')
        EXPECT_LEFT_LINE_M_MAX = ParamServer.get_value('extrapolation_lines.left_m_max')

        front_lines = MathLines()
        left_lines = MathLines()

        # for return
        front_line = None
        left_line = None

        for line in lines:
            for x1, y1, x2, y2 in line:
                wk_line = MathLine(x1, y1, x2, y2)

                if EXPECT_FRONT_LINE_M_MIN <= abs(wk_line.piangle) <= EXPECT_FRONT_LINE_M_MAX:
                    front_lines.append(wk_line)

                elif (((EXPECT_LEFT_LINE_M_MIN <= wk_line.piangle <= EXPECT_LEFT_LINE_M_MAX)
                       or (EXPECT_LEFT_LINE_M_MIN <= wk_line.piangle + 1 <= EXPECT_LEFT_LINE_M_MAX))
                      and (wk_line.x1 < (640. / 1280.) * self.img.shape[1])
                      and (wk_line.x2 < (640. / 1280.) * self.img.shape[1])):
                    # left curve
                    left_lines.append(wk_line)

        if (front_lines.get_num() > 0):
            y_min, x_min = front_lines.get_y_min()
            y_max, _x = front_lines.get_y_max()
            th = (50. / 480.) * self.img.shape[0]  # self.img.shape[0]が画像の縦
            x_max = front_lines.get_rough_x(y_max, threshold=th)
            front_line = MathLine(x_min, y_min, x_max, y_max)

        if (left_lines.get_num() > 0):
            y_min, x_min = left_lines.get_y_min()
            y_max, x_max = left_lines.get_y_max()
            left_line = MathLine(x_min, y_min, x_max, y_max)

        return front_line, left_line

    def resize(self, scale_size):
        self.img = cv2.resize(self.img, None, fx=scale_size, fy=scale_size)

    def preprocess(self):
        if ParamServer.get_value('system.color_filter'):
            self.__color_filter()
        if ParamServer.get_value('system.to_gray'):
            self.__to_gray()
        if ParamServer.get_value('system.blur'):
            self.__blur()
        if ParamServer.get_value('system.detect_edge'):
            self.__detect_edge()

    def detect_line(self, bin_out=False, color_pre=[0, 255, 0], color_final=[0, 0, 255], thickness_pre=1, thickness_final=8):
        MASK_V1 = [300. / 1280., 440. / 480.]
        MASK_V2 = [580. / 1280., 260. / 480.]
        MASK_V3 = [700. / 1280., 260. / 480.]
        MASK_V4 = [980. / 1280., 440. / 480.]

        # image mask
        if ParamServer.get_value('system.image_mask'):
            vertices = np.array([[MASK_V1, MASK_V2, MASK_V3, MASK_V4]], dtype=np.float)
            self.__mask(vertices)

        # line detect
        pre_lines = self.__houghline()
        f_line, l_line = self.__extrapolation_lines(pre_lines)

        # create image
        if len(self.img.shape) == 3:
            line_img = np.zeros((self.img.shape), np.uint8)
        else:
            line_img = np.zeros((self.img.shape[0], self.img.shape[1], 3), np.uint8)

        if not bin_out:
            # draw pre_lines
            if pre_lines is not None:
                for x1, y1, x2, y2 in pre_lines[0]:
                    cv2.line(line_img, (x1, y1), (x2, y2), color_pre, thickness_pre)

        # draw final_lines
        if bin_out:
            color_final = [255, 255, 255]

        if f_line is not None:
            cv2.line(line_img, (f_line.x1, f_line.y1), (f_line.x2, f_line.y2), color_final, thickness_final)
        if l_line is not None:
            cv2.line(line_img, (l_line.x1, l_line.y1), (l_line.x2, l_line.y2), color_final, thickness_final)

        self.img = line_img
        return f_line, l_line

    def overlay(self, img):
        ALPHA = 1.0
        BETA = 0.5
        GAMMA = 2.0
        color_img = self.get_img()
        self.img = cv2.addWeighted(color_img, ALPHA, img, BETA, GAMMA)
