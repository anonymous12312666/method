
import torch.nn as nn
from torch import cat, relu

class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        #1
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        #2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, padding=1)
        #3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        #4
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        #5
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        #6
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        #7
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv8_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv8_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        #8
        self.conv9 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)
        self.final_activation = nn.Sigmoid()




    def forward(self, x,y):
        t=cat([x,y],dim=1)
        x=self.conv1_1(x)
        x=relu(x)
        y = self.conv1_1(y)
        y = relu(y)
        t=self.conv1_2(t)
        #
        x=self.conv2(x)
        #x=laplace_filter(x)
        x = relu(x)

        y = self.conv2(y)
        #y= laplace_filter(y)
        y = relu(y)

        t = self.conv2(t)
        t = relu(t)
        t = cat([x, y,t], dim=1)
        t = self.conv2_1(t)
        #

        t1= self.conv3(t)
        t1 = relu(t1)
        t2=self.conv3(t+t1)
        t2 = relu(t2)
        t3=self.conv3(t2+t1)
        t3 = relu(t3)
        t4=self.conv3(t2+t3)
        t4 = relu(t4)
        #
        t4 = self.conv4(t4)
        t4 = relu(t4)
        #
        t5=self.conv5(t4)
        t5 = relu(t5)
        #
        t5=self.conv6(t5)
        t5 = relu(t5)
        #
        t5 = self.conv7(t5)
        t5 = relu(t5)
        #
        t5 = self.conv8_1(t5)
        t5 = relu(t5)
        #
        t5=cat([t5,t5],dim=1)

        t5 = self.conv8_2(t5)
        t5 = relu(t5)
        t5 = self.conv8_3(t5)
        t5 = relu(t5)
        #
        t5 = self.conv9(t5)
        t5 = relu(t5)
        #
        t5 = self.conv10(t5)
        t5 = relu(t5)
        #
        t5 = self.conv11(t5)
        t5 = self.final_activation(t5)
        return t5
