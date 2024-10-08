import torch
import torch.nn as nn

class Convention(nn.Module):
    def __init__(self, in_channels, out_channels, conv_size, conv_stride, padding, need_bn=True):
        super(Convention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, conv_size, conv_stride, padding, bias=False if need_bn else True)
        self.leakey_relu = nn.LeakyReLU(inplace=True, negative_slope=1e-1)
        self.need_bn = need_bn
        if need_bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.leakey_relu(self.conv(x))) if self.need_bn else self.leakey_relu(self.conv(x))

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch,nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class YOLO_Feature(nn.Module):
    def __init__(self, classes_num=80):
        super(YOLO_Feature, self).__init__()

        self.Conv_Feature = nn.Sequential(
            Convention(3, 64, 7, 2, 3),
            nn.MaxPool2d(2, 2),

            Convention(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            Convention(192, 128, 1, 1, 0),
            Convention(128, 256, 3, 1, 1),
            Convention(256, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            Convention(512, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            Convention(512, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            Convention(512, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            Convention(512, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            Convention(512, 512, 1, 1, 0),
            Convention(512, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )

        self.Conv_Semanteme = nn.Sequential(
            Convention(1024, 512, 1, 1, 0),
            Convention(512, 1024, 3, 1, 1),
            Convention(1024, 512, 1, 1, 0),
            Convention(512, 1024, 3, 1, 1),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Linear(1024, classes_num)

    def forward(self, x):
        x = self.Conv_Feature(x)
        x = self.Conv_Semanteme(x)
        x = self.avg_pool(x)
        
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.linear(x)

        return x
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, Convention):
                m.init_weight()


if __name__ == '__main__':
    x = torch.randn(1, 3, 448, 448)
    model = YOLO_Feature()
    model.init_weight()
    print(model(x).shape)