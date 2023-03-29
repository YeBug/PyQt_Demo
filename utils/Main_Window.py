# -*coding=utf-8

import logging
import os
import sys
import time
from datetime import datetime

import cv2
import numpy
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from utils import detect, general
from utils.resource import Yolo2onnx_detect_Demo_UI, resource_rc

resource_rc.qInitResources()


class StdOut(QtCore.QObject):
    """rewrite sys.Stdout"""
    signalForText = QtCore.pyqtSignal(str)

    def write(self, text):
        if text == '\n': return
        if not isinstance(text, str): text = str(text)
        if len(text) > 500: text = text[0:10] + ' ...... ' + text[-10:-1]
        self.signalForText.emit(text)

    def flush(self):
        pass

class DetectThread(QtCore.QThread):
    """检测线程"""
    img_sig = QtCore.pyqtSignal(numpy.ndarray)

    def __init__(self, model: detect.YOLOv5 = None, dataset: detect.DataLoader = None):
        super(DetectThread, self).__init__()
        self.is_pause = False
        self.is_running = False
        self.is_detecting = False
        self.model = model
        self.dataset: detect.DataLoader = dataset
        self.display_fps = True
        self.print_result = True

    def stopThread(self):
        if not self.is_running:
            return
        self.is_running = False
        self.is_detecting = False

    def stopDetect(self):
        if not self.is_detecting:
            return
        self.is_detecting = False

    def startThread(self):
        if self.is_running:
            return
        self.is_detecting = False
        self.is_running = True
        if not self.isRunning():
            self.start()

    def startDetect(self):
        if self.is_detecting:
            return
        self.is_detecting = True
        self.is_running = True
        if not self.isRunning():
            self.start()

    def pauseDetect(self):
        self.is_pause = True

    def main(self):  # 主函数
        for img, path in self.dataset:
            if not self.is_running:
                break
            res = {}
            if self.is_detecting:
                try:
                    res = self.model.detect(path)
                    self.img_sig.emit(res)
                except Exception as e:
                    print(e)
                    self.is_detecting = False
                    print('stop')

        print(f'"{self.dataset.source}" finished.')
        del self.dataset
        self.stopThread()

    def run(self) -> None:
        try:
            self.main()
        except Exception as e:
            logging.exception(e)


# noinspection PyAttributeOutsideInit
class MainWindow(QtWidgets.QMainWindow, Yolo2onnx_detect_Demo_UI.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # setup ui, connect callback
        self.setupUi(self)
        self.statusBar().showMessage('initializing...')
        self.UI()

        # init params
        self.save_video = False
        self.source = ''
        self.flip_type = (None, 1, 0, -1)
        self.rotate_type = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180)
        self.video_writer: cv2.VideoWriter
        self.box_color = (255, 0, 0)

        # install event
        self.installEventFilter(self)
        self.setAcceptDrops(True)

        # init detect thread
        self.dt = DetectThread()
        self.dt.img_sig.connect(self.displayImg)
        self.dt.finished.connect(lambda: self.statusBar().showMessage('exit'))
        self.dt.finished.connect(self.stop)
        self.dt.model = detect.YOLOv5()
        self.dt.model.initConfig()

        # self.dt.model.initModel("D:\\git\\yolov5-onnx-pyqt-exe\\need\\models\\liou_transformer.onnx")  # cv2.dnn or onnxruntime

        self.loadConfig()

        self.statusBar().showMessage('initialized', 5000)

    def UI(self):  # 槽函数
        self.toolButton.clicked.connect(self.changeOutputPath)  # 选择保存位置
        self.toolButton_2.clicked.connect(lambda: self.changeMediaFile(None))  # 选择媒体文件
        self.toolButton_3.clicked.connect(lambda: self.changeModelFile(None))  # 选择模型
        self.pushButton_4.clicked.connect(self.start)  # 开始检测槽函数
        self.pushButton_5.clicked.connect(self.stop)
        self.comboBox.currentIndexChanged.connect(self.indexChanged)  # 输入方式切换
        self.pushButton_3.clicked.connect(lambda: self.saveToFile(self.pushButton_3))  # 保存日志
        self.pushButton.clicked.connect(lambda: self.saveToFile(self.pushButton))  # 保存截图
        self.checkBox_4.clicked.connect(lambda: self.saveToFile(self.checkBox_4))  # 保存视频
        self.pushButton_2.clicked.connect(lambda: os.popen(f'explorer "{self.lineEdit.text()}"'))  # 打开保存目录
        self.textBrowser.anchorClicked.connect(lambda x: os.popen(f'"{x.toLocalFile()}"'))  # 超链接打开本地文件
        self.checkBox_5.clicked.connect(self.printResult)  # 打印检测结果
        self.checkBox_6.clicked.connect(self.changeInputConfig)  # 是否返回坐标
        self.checkBox_2.clicked.connect(self.changeInputConfig)  # 是否画锚框
        self.doubleSpinBox.valueChanged.connect(self.changeInputConfig)  # 更改置信度
        self.doubleSpinBox_2.valueChanged.connect(self.changeInputConfig)  # 更改IOU
        self.toolButton_6.clicked.connect(self.changeBoxColor)  # 更改锚框颜色
        self.pushButton_8.toggled.connect(self.lockBottom)  # 锁定切换 槽函数
        self.pushButton_10.clicked.connect(lambda: self.textBrowser.clear())  # 清空控制台
        self.pushButton_9.clicked.connect(self.resetSource)  # 重置输入源

        self.pushButton_6.setHidden(True)  # 暂停按钮, 暂时隐藏
        self.line.setHidden(True)

    def loadConfig(self):  # 加载配置
        cfg = general.cfg('config.cfg')
        self.setSource(cfg.search('root', 'input_source', default_value=0))
        self.lineEdit_3.setText(cfg.search('root', 'model_path', os.path.join('need', 'models', 'liou_transformer.onnx')))
        self.doubleSpinBox.setValue(cfg.search('root', 'conf_thres', default_value=0.5, return_type=float))
        self.doubleSpinBox_2.setValue(cfg.search('root', 'iou_thres', default_value=0.5, return_type=float))
        self.checkBox_2.setChecked(cfg.search('root', 'display_box', default_value=True, return_type=bool))
        self.changeBoxColor(eval(cfg.search('root', 'box_color', default_value='(255,0,0)')))
        self.checkBox_5.setChecked(cfg.search('root', 'print_result', default_value=True, return_type=bool))
        self.checkBox_6.setChecked(cfg.search('root', 'with_pos', default_value=False, return_type=bool))
        self.checkBox_4.setChecked(cfg.search('root', 'record_video', default_value=False, return_type=bool))
        self.spinBox.setValue(cfg.search('root', 'record_fps', default_value=15, return_type=int))
        self.lineEdit.setText(cfg.search('root', 'out_path', default_value=os.path.join(os.getcwd(), 'out')))
        if cfg.search('root', 'detect_status', default_value=False, return_type=bool):
            self.start()

    def saveConfig(self):  # 保存配置
        with general.cfg('config.cfg') as cfg:
            cfg.set('root', 'detect_status', self.dt.is_detecting)
            cfg.set('root', 'input_source', self.source)
            cfg.set('root', 'model_path', self.lineEdit_3.text())
            cfg.set('root', 'conf_thres', self.doubleSpinBox.value())
            cfg.set('root', 'iou_thres', self.doubleSpinBox_2.value())
            cfg.set('root', 'display_box', self.checkBox_2.isChecked())
            cfg.set('root', 'box_color', self.box_color)
            cfg.set('root', 'print_result', self.checkBox_5.isChecked())
            cfg.set('root', 'with_pos', self.checkBox_6.isChecked())
            cfg.set('root', 'record_video', self.checkBox_4.isChecked())
            cfg.set('root', 'record_fps', self.spinBox.value())
            cfg.set('root', 'out_path', self.lineEdit.text())

    def changeInputConfig(self):  # 更改输入配置
        if self.dt.model is not None:
            self.dt.model.conf_threshold = self.doubleSpinBox.value()  # 更改置信度
            self.dt.model.iou_threshold = self.doubleSpinBox_2.value()  # 更改IOU
            self.dt.model.draw_box = self.checkBox_2.isChecked()  # 是否显示锚框
            self.dt.model.with_pos = self.checkBox_6.isChecked()  # 是否返回坐标

    def changeBoxColor(self, color: tuple = None):  # 更改锚框颜色
        if not color:
            old_color = self.dt.model.box_color if self.dt.model else self.box_color
            new_color = QtWidgets.QColorDialog.getColor(QtGui.QColor(*old_color))
            if not new_color.isValid():
                return
            color = new_color.getRgb()[0:3]
        self.toolButton_6.setStyleSheet(f"color:rgb{color}")
        self.box_color = color
        if self.dt.model is None:
            return
        self.dt.model.box_color = self.box_color
        self.dt.model.txt_color = tuple(255 - x for x in self.box_color)

    def printResult(self):  # 打印检测结果
        self.dt.print_result = self.checkBox_5.isChecked()

    def saveToFile(self, index):  # 保存截图、视频、日志
        os.makedirs(self.lineEdit.text(), exist_ok=True)
        head = datetime.now().strftime('%m-%d %H-%M-%S')
        # 保存截图
        if index == self.pushButton:
            path = os.path.join(self.lineEdit.text(), f'ScreenShot_{head}.png')
            if self.label.pixmap() is None:
                return
            self.label.pixmap().toImage().save(path)
            print(f'ScreenShot has been saved to <a href="file:///{path}">{path}</a>')

    def resetSource(self):  # 重置输入源
        if self.dt.is_detecting:
            return
        self.indexChanged(self.comboBox.currentIndex())

    def indexChanged(self, index):  # 切换输入方式
        self.dt.blockSignals(True)
        self.dt.stopThread()
        self.dt.wait()
        if index == 0:  # webcam 0
            self.setSource('0')
        elif index == 1 and os.path.exists(self.lineEdit_2.text()):  # file
            self.setSource(self.lineEdit_2.text())
        elif index == 2 or index == 4:  # url or custom data
            text, flag = QtWidgets.QInputDialog.getText(self, 'Custom Source', 'input:', text=self.source)
            if not flag:
                return
            self.setSource(text)
        elif index == 3:  # full screen 0
            self.setSource('screen')
        self.dt.blockSignals(False)

    def setSource(self, source, **kwargs) -> bool:  # 设置输入源
        self.source = str(source)
        try:
            self.dt.dataset = detect.DataLoader(self.source,
                                                frame_skip=-1,
                                                flip=None,
                                                rotate=None,
                                                **kwargs)
        except Exception as e:
            self.label.setText(str(e))
            self.displayLog(str(e), color='red')
            return False
        self.comboBox.blockSignals(True)
        index = (self.source == '0',
                 self.dt.dataset.is_image or self.dt.dataset.is_video,
                 self.dt.dataset.is_url,
                 self.source.lower() == 'screen',
                 True).index(True)
        self.comboBox.setCurrentIndex(index)
        self.comboBox.blockSignals(False)
        if self.dt.dataset.is_wabcam or self.dt.dataset.is_url or self.dt.dataset.is_screen:
            self.dt.startThread()
        elif self.dt.dataset.is_image or self.dt.dataset.is_video:
            self.lineEdit_2.setText(self.source)
            vc = cv2.VideoCapture(self.source)
            img = vc.read()[1]
            self.displayImg(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            vc.release()
        return 'dataset' in self.dt.__dict__.keys()

    def start(self):  # 启动检测线程
        if self.dt.is_detecting:
            print('already running')
            return
        if not os.path.exists(self.lineEdit_3.text()):
            self.displayLog(f'"{self.lineEdit_3.text()}" not exist', color='red')
            return
        if self.comboBox.currentIndex() == 1 and not os.path.exists(self.lineEdit_2.text()):  # 视频
            self.displayLog(f'"{self.lineEdit_2.text()}" not exist', color='red')
            return
        if 'dataset' not in self.dt.__dict__.keys() and not self.setSource(self.source):
            return

        self.dt.startDetect()
        self.saveToFile(self.checkBox_4)
        print('start detect')
        self.statusBar().showMessage('start detect...', 5000)

    # todo
    def pause(self):  # 暂停
        pass

    def stop(self):  # 停止检测
        if self.dt.is_detecting:
            self.dt.stopDetect()
            self.statusBar().showMessage('stop detect', 5000)
            print('stop detect')
            return
        if self.dt.is_running:
            self.dt.stopThread()
            self.dt.wait()

    def changeModelFile(self, path=None):  # 选择权重文件
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "选择模型",
                                                  os.path.abspath(self.lineEdit_3.text()),
                                                  '*.onnx')
        if path:
            self.lineEdit_3.setText(path)

    def changeMediaFile(self, path=None):  # 选择媒体文件
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件",
                                                  os.path.abspath(self.lineEdit_2.text()),
                                                  '*.asf *.avi *.gif *.m4v *.mkv *.mov *.mp4 *.mpeg *.mpg *.ts *.wmv '
                                                  '*.bmp *.dng *.jpeg *.jpg *.mpo *.png *.tif *.tiff *.webp *.pfm')
        if path:
            self.setSource(path)

    def changeOutputPath(self):  # 选择保存位置
        file_path = QFileDialog.getExistingDirectory(self, "选择保存位置", self.lineEdit.text())
        if file_path:
            self.lineEdit.setText(file_path)

    def displayImg(self, img: numpy.ndarray):  # 显示图片到label
        if self.save_video:
            self.video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QtGui.QImage.Format_RGB888)
        p = min(self.label.width() / img.width(), self.label.height() / img.height())
        pix = QtGui.QPixmap(img).scaled(int(img.width() * p), int(img.height() * p))
        self.label.setPixmap(pix)

    def displayLog(self, text: str, color='black', plain_text=False):  # 输出控制台信息到textBrowser
        head_ = f"{datetime.now().strftime('%H:%M:%S.%f')} >> "

        if text.startswith(('<',)) or plain_text:
            self.textBrowser.setTextColor(QtGui.QColor('black'))
            self.textBrowser.append(head_)
            self.textBrowser.setTextColor(QtGui.QColor(color))
            self.textBrowser.insertPlainText(text)
        else:
            text = f"{head_}<font color='{color}'>{text}"
            self.textBrowser.append(text)
        # 自动切换锁定状态
        scrollbar = self.textBrowser.verticalScrollBar()
        self.pushButton_8.setChecked(scrollbar.value() >= scrollbar.maximum())
        if self.pushButton_8.isChecked():
            scrollbar.setValue(scrollbar.maximum())

    def lockBottom(self, status):  # 锁定底部切换
        scrollbar = self.textBrowser.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum() if status else scrollbar.maximum() - 1)

    def eventFilter(self, objwatched, event):  # 重写事件过滤
        eventType = event.type()
        # 设置/取消焦点
        if eventType == QtCore.QEvent.MouseButtonPress:
            self.setFocus()
        # 快捷键: ctrl+r开始; ctrl+e停止; ctrl+s截图
        if eventType == QtCore.QEvent.KeyPress:
            if event.modifiers() == QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_R:
                self.start()
            elif event.modifiers() == QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_E:
                self.stop()
            elif event.modifiers() == QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_S:
                self.saveToFile(self.pushButton)
        # 捕获拖入文件
        if eventType == QtCore.QEvent.DragEnter:
            event.accept()
        # 判断并设置拖入文件路径
        if eventType == QtCore.QEvent.Drop:
            for file in event.mimeData().urls():
                f: str = file.toLocalFile()
                if not os.path.isfile(f):
                    return False
                if f.lower().endswith('.onnx'):
                    self.changeModelFile(f)
                elif f.lower().endswith(detect.DataLoader.IMAGE_TYPE + detect.DataLoader.VIDEO_TYPE):
                    self.changeMediaFile(f)
                elif f.lower().endswith(('.py', '.pyw')):
                    self.changePyFile(f)
        # 关闭窗口事件
        if eventType == QtCore.QEvent.Close:
            if self.dt.is_detecting:
                msgbox = QMessageBox.question(self,
                                              self.windowTitle(),
                                              '正在运行\n是否停止并关闭?',
                                              QMessageBox.Yes | QMessageBox.Ignore | QMessageBox.No,
                                              QMessageBox.Yes)
                if msgbox == QMessageBox.Yes:
                    self.stop()
                elif msgbox == QMessageBox.No:
                    event.ignore()
                    return True
            self.saveConfig()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        return super().eventFilter(objwatched, event)

    """
    res_data: 属性, 检测结果, dict类型
    img_data: 属性, 实时图片, numpy.ndarray类型
    setDetectStatus(bool): 方法, 设置检测状态, 接受一个bool类型参数, 为True时开启检测, 为False时停止检测
    setModelConfig(**kwargs): 方法, 设置模型配置, 接受关键字参数, 可选关键字参数有
        model_path(str),
        input_width(int),
        input_height(int),
        draw_box(bool),
        box_color(tuple[int,int,int]),
        txt_color(tuple[int,int,int]),
        thickness(int),
        conf_thres(float),
        iou_thres(float),
        class_names(list[str])
    """
    start_sig = QtCore.pyqtSignal(bool)

    @property
    def help(self):
        return self.__doc__

    def setDetectStatus(self, status: bool):
        if status and not self.thread.is_detecting:
            self.start_sig.emit(status)
        elif not status and self.thread.is_detecting:
            self.start_sig.emit(status)

    def setModelConfig(self, **kwargs):
        if 'model_path' in kwargs:
            self.thread.model.initModel(kwargs['model_path'])
        if self.thread.model:
            self.thread.model.__dict__.update(**kwargs)
