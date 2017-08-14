#!/usr/bin/env python
# # -*- coding: utf-8 -*-

# [how to use]
# python ros_line_detect.py image:=/image_raw


import sys
import math
import copy
import rospy
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from PyQt4.QtGui import *
from PyQt4.QtCore import *

import image_process
import setting_gui
from param_server import ParamServer


class RcLineDetect():

    def __init__(self):
        self.__thin_cnt = 0
        self.last_image_msg = None
        self.__cv_bridge = CvBridge()
        image_node = rospy.get_param("~image", "/usb_cam/image_raw")
        self.__sub = rospy.Subscriber(image_node, Image, self.callback, queue_size=1)
        self.__pub = rospy.Publisher('image_processed', Image, queue_size=1)
        ParamServer.add_cb_value_changed(self.redraw)

    def image_process(self, image_msg):
        cv_image = self.__cv_bridge.imgmsg_to_cv2(image_msg, 'bgr8')

        pre_img = image_process.ProcessingImage(cv_image)

        # pre  :処理負荷軽減のための事前縮小
        # final:deep learning学習データ用の縮小。 pre_resize * final_resizeの値が最終データとなる
        pre_scale = 1.0 / ParamServer.get_value('system.pre_resize')
        final_scale = 1.0 / ParamServer.get_value('system.final_resize')
        pre_img.resize(pre_scale)

        # 抽象化
        pre_img.preprocess()

        # 画像出力 for rqt_image_view
        if (rospy.get_param("~gui", True)):
            out_img = copy.deepcopy(pre_img)
            # 直線検出
            if ParamServer.get_value('system.detect_line'):
                out_img.detect_line()
                out_img.overlay(pre_img.get_img())

            out_img.resize(final_scale)
            self.__pub.publish(self.__cv_bridge.cv2_to_imgmsg(out_img.get_img(), 'bgr8'))

        # bin出力 for Tensorflow
        else:
            pro_img = copy.deepcopy(pre_img)
            pro_img.resize(final_scale)
            pro_array = np.reshape(pro_img.get_grayimg(), (60, 160, 1))

            out_dim = rospy.get_param("~dim", 2)
            if out_dim == 1:
                self.__pub.publish(self.__cv_bridge.cv2_to_imgmsg(pro_array, 'mono8'))

            elif out_dim == 2:
                # line detect
                line_img = copy.deepcopy(pre_img)
                f_line, l_line = line_img.detect_line(bin_out=True, thickness_final=16)

                # set line info to info_array
                info_array = np.zeros((60, 160, 1), np.uint8)

                if f_line is not None:
                    info_array[0, 0, 0] = f_line.x1 * (255. / line_img.img.shape[1])
                    info_array[0, 1, 0] = f_line.y1 * (255. / line_img.img.shape[0])
                    info_array[0, 2, 0] = f_line.x2 * (255. / line_img.img.shape[1])
                    info_array[0, 3, 0] = f_line.y2 * (255. / line_img.img.shape[0])
                if l_line is not None:
                    info_array[1, 0, 0] = l_line.x1 * (255. / line_img.img.shape[1])
                    info_array[1, 1, 0] = l_line.y1 * (255. / line_img.img.shape[0])
                    info_array[1, 2, 0] = l_line.x2 * (255. / line_img.img.shape[1])
                    info_array[1, 3, 0] = l_line.y2 * (255. / line_img.img.shape[0])

                # format for output
                line_img.resize(final_scale)
                line_array = np.reshape(line_img.get_grayimg(), (60, 160, 1))

                # output
                out_bin = np.dstack((pro_array, line_array, info_array))
                self.__pub.publish(self.__cv_bridge.cv2_to_imgmsg(out_bin, 'bgr8'))

    def redraw(self):
        if self.last_image_msg is not None:
            self.image_process(self.last_image_msg)

    def callback(self, image_msg):
        self.last_image_msg = image_msg

        # 処理性能で調整する間引き
        self.__thin_cnt += 1
        if self.__thin_cnt < ParamServer.get_value('system.thinning'):
            return
        else:
            self.__thin_cnt = 0
        self.image_process(image_msg)

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('rc_line_detect')
    gui_mode = rospy.get_param("~gui", True)
    out_dim = rospy.get_param("~dim", 2)

    thinning = rospy.get_param("~thinning", 1)
    ParamServer.set_value('system.thinning', thinning)

    if out_dim == 1:
        ParamServer.set_value('system.to_gray', 0)
        ParamServer.set_value('system.detect_line', 0)
        ParamServer.set_value('system.final_resize', 4)
        ParamServer.set_value('system.mono_output', 1)

    elif out_dim == 2:
        ParamServer.set_value('system.to_gray', 1)
        ParamServer.set_value('system.detect_line', 1)
        ParamServer.set_value('system.final_resize', 4)
        ParamServer.set_value('system.mono_output', 0)

    process = RcLineDetect()

    if (gui_mode):
        ParamServer.set_value('system.final_resize', 1)
        app = QApplication(sys.argv)
        gui = setting_gui.SettingWindow()
        app.exec_()
    else:
        process.main()
