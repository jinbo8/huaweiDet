import torch.nn as nn
import torch
import torch.nn.functional as F


class myNet_ocr(nn.Module):
    def __init__(self, cfg=None, num_classes=78, export=False, color_num=None):
        super(myNet_ocr,  self).__init__()
        if cfg is None:
            cfg =[32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256]
            # cfg =[32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256]
        self.feature = self.make_layers(cfg,  True)
        self.export = export
        self.color_num= color_num
        if self.color_num:
            self.gap = nn.AdaptiveAvgPool2d(output_size=1)
            self.color_classifier=nn.Conv2d(cfg[-1], 5, kernel_size=1, stride=1)
            self.color_bn = nn.BatchNorm2d(5)
            self.flatten = nn.Flatten()
            # self.relu = nn.ReLU(inplace=True)
        # self.classifier = nn.Linear(cfg[-1],  num_classes)
        # self.loc=nn.MaxPool2d((2,  2),  (5,  1),  (0,  1), ceil_mode=True)
        # self.loc = nn.AvgPool2d((2,  2),  (5,  2),  (0,  1), ceil_mode=False)
        self.loc = nn.MaxPool2d((5,  2),  (1,  1), (0, 1), ceil_mode=False)
        self.newCnn = nn.Conv2d(cfg[-1], num_classes, 1, 1)
        # self.newBn = nn.BatchNorm2d(num_classes)
    def make_layers(self,  cfg,  batch_norm=False):
        layers = []
        in_channels = 3
        for i in range(len(cfg)):
            if i == 0:
                conv2d = nn.Conv2d(in_channels,  cfg[i],  kernel_size=5, stride =1)
                if batch_norm:
                    layers += [conv2d,  nn.BatchNorm2d(cfg[i]),  nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d,  nn.ReLU(inplace=True)]
                in_channels = cfg[i]
            else :
                if cfg[i] == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3,  stride=2, ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels,  cfg[i],  kernel_size=3,  padding=(1, 1), stride =1)
                    if batch_norm:
                        layers += [conv2d,  nn.BatchNorm2d(cfg[i]),  nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d,  nn.ReLU(inplace=True)]
                    in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.color_num:
            x_color = self.color_classifier(x)
            x_color = self.color_bn(x_color)
            # x_color=self.relu(x_color)
            x_color = self.gap(x_color)
            x_color = self.flatten(x_color)
        x = self.loc(x)
        x = self.newCnn(x)
       
        if self.export:
            conv = x.squeeze(2) # b *512 * width
            conv = conv.transpose(2, 1)  # [w,  b,  c]
            if self.color_num:
                return conv, x_color
            return conv
        else:
            b,  c,  h,  w = x.size()
            assert h == 1,  "the height of conv must be 1"
            conv = x.squeeze(2) # b *512 * width
            conv = conv.permute(2,  0,  1)  # [w,  b,  c]
            output = F.log_softmax(conv,  dim=2)
            if self.color_num:
                return output, x_color
            return output


class myNet_ocr_color(nn.Module):
    def __init__(self, cfg=None, num_classes=78, export=False, color_num=None):
        super(myNet_ocr_color,  self).__init__()
        if cfg is None:
            cfg = [32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256]
            # cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256]
        self.feature = self.make_layers(cfg,  True)
        self.export = export
        self.color_num = color_num
        self.conv_out_num = 12  #颜色第一个卷积层输出通道12
        if self.color_num:
            self.conv1 = nn.Conv2d(cfg[-1], self.conv_out_num, kernel_size=3, stride=2)
            self.bn1 = nn.BatchNorm2d(self.conv_out_num)
            self.relu1 = nn.ReLU(inplace=True)
            self.gap  = nn.AdaptiveAvgPool2d(output_size=1)
            self.color_classifier = nn.Conv2d(self.conv_out_num, self.color_num, kernel_size=1, stride=1)
            self.color_bn = nn.BatchNorm2d(self.color_num)
            self.flatten = nn.Flatten()
            # self.relu = nn.ReLU(inplace=True)
        # self.classifier = nn.Linear(cfg[-1],  num_classes)
        # self.loc=nn.MaxPool2d((2,  2),  (5,  1),  (0,  1), ceil_mode=True)
        # self.loc=nn.AvgPool2d((2,  2),  (5,  2),  (0,  1), ceil_mode=False)
        self.loc = nn.MaxPool2d((5,  2),  (1,  1), (0, 1), ceil_mode=False)
        self.newCnn = nn.Conv2d(cfg[-1], num_classes, 1, 1)
        # self.newBn = nn.BatchNorm2d(num_classes)

    def make_layers(self,  cfg,  batch_norm=False):
        layers = []
        in_channels = 3
        for i in range(len(cfg)):
            if i == 0:
                conv2d = nn.Conv2d(in_channels,  cfg[i],  kernel_size=5, stride =1)
                if batch_norm:
                    layers += [conv2d,  nn.BatchNorm2d(cfg[i]),  nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d,  nn.ReLU(inplace=True)]
                in_channels = cfg[i]
            else :
                if cfg[i] == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3,  stride=2, ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels,  cfg[i],  kernel_size=3,  padding=(1, 1), stride =1)
                    if batch_norm:
                        layers += [conv2d,  nn.BatchNorm2d(cfg[i]),  nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d,  nn.ReLU(inplace=True)]
                    in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self,  x):
        x = self.feature(x)
        if self.color_num:
            x_color = self.conv1(x)
            x_color = self.bn1(x_color)
            x_color = self.relu1(x_color)
            x_color = self.color_classifier(x_color)
            x_color = self.color_bn(x_color)
            x_color = self.gap(x_color)
            x_color = self.flatten(x_color) 
        x = self.loc(x)
        x = self.newCnn(x)
       
        if self.export:
            conv = x.squeeze(2) # b *512 * width
            conv = conv.transpose(2, 1)  # [w,  b,  c]
            if self.color_num:
                return conv, x_color
            return conv
        else:
            b,  c,  h,  w = x.size()
            assert h == 1,  "the height of conv must be 1"
            conv = x.squeeze(2) # b *512 * width
            conv = conv.permute(2,  0,  1)  # [w,  b,  c]
            output = F.log_softmax(conv,  dim=2)
            if self.color_num:
                return output, x_color
            return output


if __name__ == '__main__':
    x = torch.randn(1, 3, 48, 168)
    cfg = [32, 'M', 64, 'M', 128, 'M', 256]
    # ckpt = torch.load(r"color_model2/0.9925233644859813_epoth_60_model.pth")
    # cfg = ckpt['cfg']
    model = myNet_ocr_color(num_classes=78, export=True, cfg=cfg, color_num=5)
    # model.load_state_dict(ckpt['state_dict'], strict=False)
    # for key, value in model.named_parameters():
    #     print(key)
    print(model)
    # model.eval()
    out, color= model(x)
    print(out.shape)
    print(color.shape)