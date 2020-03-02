import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import pdb
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class convnet(nn.Module):
    def __init__(self,num_classes=1):
        super(convnet,self).__init__()
       
        self.bn0     = nn.BatchNorm2d(3)
        self.conv1   = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2   = nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1)
        self.conv3   = nn.Conv2d(32,64, kernel_size=3, stride=2, padding=1)
        self.conv4   = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AvgPool2d(32, stride=1)
        self.fc      = nn.Linear(64, num_classes)
        
        self.sigmoid    = nn.Sigmoid()

    def get_feature(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 28x28
        x = self.maxpool(x) # 14x14

        x = self.conv2(x)
        x = self.relu(x) #14x14
        
        x = self.conv3(x)
        x = self.relu(x) # 7x7
        x = self.conv4(x)
        x = self.relu(x) # 7x7
        feat_low = x

        return feat_low

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.relu(x)  # 28x28
        x = self.maxpool(x) # 14x14

        x = self.conv2(x)
        x = self.relu(x) #14x14
        feat_out = x  
        x = self.conv3(x)
        x = self.relu(x) # 7x7
        x = self.conv4(x)
        x = self.relu(x) # 7x7
        feat_low = x
        feat_low = self.avgpool(feat_low)
        feat_low = feat_low.view(feat_low.size(0),-1)
        # import pdb; pdb.set_trace()
        y_low = self.fc(feat_low)
        # y_low = y_low.squeeze()
        y_low = self.sigmoid(y_low)

        return feat_out, y_low


class Predictor(nn.Module):
    def __init__(self, input_ch=32, num_classes=1):
        super(Predictor, self).__init__()
        self.pred_conv1 = nn.Conv2d(input_ch, input_ch, kernel_size=3,
                                    stride=1, padding=1)
        self.pred_bn1   = nn.BatchNorm2d(input_ch)
        self.relu       = nn.ReLU(inplace=True)
        self.pred_conv2 = nn.Conv2d(input_ch, num_classes, kernel_size=3,
                                    stride=1, padding=1)
        

        self.sigmoid    = nn.Sigmoid()
        # self.sigmoid    = nn.Sigmoid()
    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.pred_conv1(x)
        x = self.pred_bn1(x)
        x = self.relu(x)
        x = self.pred_conv2(x)
        # x = self.fc(x)
        #px = x
        px = self.sigmoid(x)
        
        return x,px

