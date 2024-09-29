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
