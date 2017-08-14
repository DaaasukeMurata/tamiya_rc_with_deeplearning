#!/usr/bin/env python
# coding: UTF-8

import os
import math
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import tensorflow as tf

import rospy
from std_msgs.msg import UInt16MultiArray
from sensor_msgs.msg import Image

from model import CNNModel


# define
CKPT_PATH = os.path.abspath(os.path.dirname(__file__)) + '/ckpt/'
IMG_HEIGHT = 60
IMG_WIDTH = 160
IMG_DIM = 2


class LineInfo(object):

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.piangle = self.__get_piangle()

    def __get_piangle(self):
        if self.x1 == 0:
            return 0.
        # y = tan(θ) * x + b
        vy = int(self.y2) - int(self.y1)
        vx = int(self.x2) - int(self.x1)
        return math.atan2(vy, vx) / math.pi


class RcImageSteer():

    def __init__(self):
        rospy.init_node('rc_image2steer')

        # --- for Tensorflow ---
        self.cnn = CNNModel()
        model_name = rospy.get_param("~model_name")
        self.cnn.saver.restore(self.cnn.sess, CKPT_PATH + model_name)

        # --- for ROS ---
        self.adj_steer = rospy.get_param("~line_adjust_steer", 20)

        self._cv_bridge = CvBridge()
        test_mode = rospy.get_param("~testmode", False)
        if test_mode:
            self._pub = rospy.Publisher('servo2', UInt16MultiArray, queue_size=1)
        else:
            self._pub = rospy.Publisher('servo', UInt16MultiArray, queue_size=1)
        self._sub = rospy.Subscriber('image_processed', Image, self.callback, queue_size=1)
        print 'RcImageSteer init done.'

    # # for model ver3
    # def steer_by_model(self, image, f_line, l_line):
    #     img = image
    #     line = np.array([f_line.x1, f_line.y1, f_line.x2, f_line.y2, f_line.piangle,
    #                      l_line.x1, l_line.y1, l_line.x2, l_line.y2, l_line.piangle])
    #     line = line.reshape([1, 10])

    #     p = self.cnn.sess.run(self.cnn.predictions,
    #                           feed_dict={self.cnn.image_holder: img,
    #                                      self.cnn.line_meta_holder: line,
    #                                      self.cnn.keepprob_holder: 1.0})
    #     answer = np.argmax(p, 1)
    #     # print answer[0], p[0, answer[0]]
    #     return answer[0]

    # for model ver2
    def steer_by_model(self, image, f_line, l_line):
        img = image
        p = self.cnn.sess.run(self.cnn.predictions,
                              feed_dict={self.cnn.input_holder: img,
                                         self.cnn.keepprob_holder: 1.0})
        answer = np.argmax(p, 1)
        # print answer[0], p[0, answer[0]]
        return answer[0]

    def steer_by_line(self, f_line):
        if f_line.x1 == 0:
            return None
        dif = (0.5 - f_line.piangle) * 10 * self.adj_steer
        if (((f_line.x1 + f_line.x2 < 255) and (f_line.piangle < 0.5)) or ((f_line.x1 + f_line.x2 > 255) and (f_line.piangle > 0.5))):
            dif = dif * -1
        steer = 90 + dif
        if steer < 30:
            steer = 30
        elif steer > 150:
            steer = 150
        # print f_line.piangle, steer
        return steer

    def callback(self, image_msg):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # dim1とdim2が[height, width, depth]の画像、dim3はLineMetaInfo
        dim1, dim2, lines = np.dsplit(cv_image, 3)
        image = np.dstack((dim1, dim2))
        image = np.reshape(image, (1, IMG_HEIGHT, IMG_WIDTH, IMG_DIM))
        f_line = LineInfo(int(lines[0, 0, 0]), int(lines[0, 1, 0]), int(lines[0, 2, 0]), int(lines[0, 3, 0]))
        l_line = LineInfo(int(lines[1, 0, 0]), int(lines[1, 1, 0]), int(lines[1, 2, 0]), int(lines[1, 3, 0]))

        steer = None

        line_trace = rospy.get_param("~use_line_trace", False)
        if line_trace:
            steer = self.steer_by_line(f_line)

        if steer is None:
            steer = self.steer_by_model(image, f_line, l_line)

        a = UInt16MultiArray()
        a.data = [steer, 83]
        self._pub.publish(a)

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    process = RcImageSteer()
    process.main()
