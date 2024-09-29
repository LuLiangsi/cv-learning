import torch
import torch.nn as nn

class YOLO_Loss(nn.Module):
    def __init__(self, S=7, B=2, Classes=20, l_coord=5, pos_conf=1, pos_cls=1, l_noobj=0.5):
        super(YOLO_Loss, self).__init__()
        self.S = S
        self.B = B
        self.Classes = Classes
        self.l_coord = l_coord
        self.pos_conf = pos_conf
        self.pos_cls = pos_cls
        self.l_noobj = l_noobj

    def iou_force(self, bounding_box, ground_box, gridX, gridY, img_size=448, grid_size=64):
        # 计算两个box的IoU值
        # predict_box: [centerX, centerY, width, height]
        # ground_box : [centerX / self.grid_cell_size - indexJ,centerY / self.grid_cell_size - indexI,(xmax-xmin)/self.img_size,(ymax-ymin)/self.img_size,1,xmin,ymin,xmax,ymax,(xmax-xmin)*(ymax-ymin)
        # 1.  预处理 predict_box  变为  左上X,Y  右下X,Y  两个边界点的坐标 避免浮点误差 先还原成整数
        # 不要共用引用
        # [xmin,ymin,xmax,ymax]

        predict_box = list([0,0,0,0])
        

