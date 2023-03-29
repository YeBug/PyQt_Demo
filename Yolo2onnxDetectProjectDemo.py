# -*coding=utf-8

import argparse
import cgitb
import logging
import os
import sys
import time

import cv2
from PyQt5 import QtWidgets, QtCore
from utils.Main_Window import MainWindow, StdOut

def run(**kwargs):


    # init GUI
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 高分辨率
    app = QtWidgets.QApplication(sys.argv)
    mainwindow = MainWindow()
    
    # redirect stdout
    stdout = StdOut()
    stdout.signalForText.connect(mainwindow.displayLog)
    sys.stdout = stdout
    sys.stderr = stdout
    logging.StreamHandler(stdout)

    # show main window
    mainwindow.show()
    sys.exit(app.exec_())

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nogui', action='store_true')
    parser.add_argument('--weights', type=str, default='need/models/liou_transformer.onnx')
    parser.add_argument('--classes', type=str, default='need/yolov7-tiny.txt')
    parser.add_argument('--source', type=str, default='data')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640], help='inference size w,h')
    parser.add_argument('--conf_thres', type=float, default=0.5)
    parser.add_argument('--iou_thres', type=float, default=0.5)
    parser.add_argument('--save_path', type=str, default='out')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--video_split', action='store_true')
    opt_ = parser.parse_args()
    opt_.imgsz *= 2 if len(opt_.imgsz) == 1 else 1  # expand
    return opt_


if __name__ == "__main__":
    # dump logs
    log_dir = os.path.join(os.getcwd(), 'log')
    os.makedirs(log_dir, exist_ok=True)
    cgitb.enable(format='text', logdir=log_dir)

    opt = parse_opt()
    run(**vars(opt))
