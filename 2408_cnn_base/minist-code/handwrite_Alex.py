#########################################
#
#   need to train the model first
#   our model params is named net_alex_256.params
#
#########################################
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint

import torch
from torch import nn
import numpy as np
import os

net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10))

class PaintApp(QWidget):
    def __init__(self, pen_width):
        super().__init__()
        self.initUI()
        self.pen_width = pen_width

    def initUI(self):
        self.setWindowTitle('HandWrite Mumbers')
        self.setGeometry(100, 100, 260, 320)

        self.canvas = QImage(224, 224, QImage.Format_RGB32)
        self.canvas.fill(Qt.black)

        self.label = QLabel(self)
        self.label.setPixmap(QPixmap.fromImage(self.canvas))

        # 添加用于显示预测值的 QLabel
        self.prediction_label = QLabel('Prediction: None', self)
        self.prediction_label.setAlignment(Qt.AlignCenter)

        self.clear_button = QPushButton('Clear', self)
        self.clear_button.clicked.connect(self.clear)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.prediction_label)
        self.prediction_label.setStyleSheet("font-size: 22px;")  # 设置字体大小
        layout.addWidget(self.clear_button)
        self.setLayout(layout)

        self.drawing = False
        self.last_point = QPoint()
        self.offset = QPoint(20, 20)  # 偏移量

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos() - self.offset

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton & self.drawing:
            painter = QPainter(self.canvas)
            painter.setPen(QPen(Qt.white, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            current_point = event.pos() - self.offset
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update_canvas()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            scaled_canvas = self.canvas.scaled(224, 224, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

            # 转换为灰度图像
            gray_canvas = scaled_canvas.convertToFormat(QImage.Format_Grayscale8)
            
            # 将图像数据转换为 NumPy 数组
            width = gray_canvas.width()
            height = gray_canvas.height()
            ptr = gray_canvas.bits()
            ptr.setsize(gray_canvas.byteCount())
            arr = torch.from_numpy(np.array(ptr).reshape(height, width))
            arr = arr.unsqueeze(0).unsqueeze(0) 
            arr = arr / 255.0
            with torch.no_grad():
                # for layer in net:
                #     arr=layer(arr)
                #     print(layer.__class__.__name__,'output shape:\t',arr.shape)
                predictions = net(arr)
                predicted_labels = torch.argmax(predictions, axis=1)
                self.prediction_label.setText(f'Prediction: {predicted_labels.item()}')
            

    def update_canvas(self):
        self.label.setPixmap(QPixmap.fromImage(self.canvas))

    def clear(self):
        self.canvas.fill(Qt.black)
        self.update_canvas()
        self.prediction_label.setText('Prediction: None')

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    net.load_state_dict(torch.load(os.path.join(dir_path, 'net_alex_256.params'), map_location=torch.device('cpu')))

    app = QApplication(sys.argv)
    window = PaintApp(pen_width=20)
    window.show()
    sys.exit(app.exec_())