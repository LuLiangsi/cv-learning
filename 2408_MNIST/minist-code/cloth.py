import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from d2l import torch as d2l

class MPLScratch(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_inputs, num_hiddens), 
                                 nn.ReLU(), 
                                 nn.Linear(num_hiddens, num_outputs))
    def forward(self, X):
        return self.net(X.reshape(-1, 784))

text_labels = ['T恤', '裤子', '套头衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '踝靴']

class PaintApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('HandWrite Mumbers')
        self.setGeometry(100, 100, 280, 320)

        self.canvas = QImage(280, 280, QImage.Format_RGB32)
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
            painter.setPen(QPen(Qt.white, 30, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            current_point = event.pos() - self.offset
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update_canvas()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            scaled_canvas = self.canvas.scaled(28, 28, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

            # 转换为灰度图像
            gray_canvas = scaled_canvas.convertToFormat(QImage.Format_Grayscale8)
            
            # 将图像数据转换为 NumPy 数组
            width = gray_canvas.width()
            height = gray_canvas.height()
            ptr = gray_canvas.bits()
            ptr.setsize(gray_canvas.byteCount())
            arr = torch.from_numpy(np.array(ptr).reshape(height, width))
            arr = arr / 255.0
            
            model.eval()
            with torch.no_grad():
                predictions = model(arr)
                predicted_labels = torch.argmax(predictions, axis=1)
                
                self.prediction_label.setText(f'Prediction: {text_labels[predicted_labels.item()]}')
            

    def update_canvas(self):
        self.label.setPixmap(QPixmap.fromImage(self.canvas))

    def clear(self):
        self.canvas.fill(Qt.black)
        self.update_canvas()
        self.prediction_label.setText('Prediction: None')

if __name__ == '__main__':

    model = MPLScratch(784, 10, 256, 0.1)
    model.load_state_dict(torch.load('model_cloth.pth'))

    app = QApplication(sys.argv)
    window = PaintApp()
    window.show()
    sys.exit(app.exec_())