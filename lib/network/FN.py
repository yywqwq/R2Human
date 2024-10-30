import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Norm2d = nn.BatchNorm2d # lambda planes: nn.GroupNorm(planes//16, planes)

class BTNK(nn.Module):
    def __init__(self, inplanes, planes, outplanes):
        super(BTNK, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            Norm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            Norm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, outplanes, kernel_size=1, bias=False),
            Norm2d(outplanes)
        )
        self.res = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
            Norm2d(outplanes),
        ) if inplanes!=outplanes else nn.Identity()

    def forward(self, x):
        return F.relu(self.res(x) + self.net(x), inplace=True)

class RES(nn.Module):
    def __init__(self, planes):
        super(RES, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            Norm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            Norm2d(planes)
        )

    def forward(self, x):
        return F.relu(x + self.net(x), inplace=True)

def make_RES4(planes):
    return nn.Sequential(
        RES(planes), RES(planes), RES(planes), RES(planes)
    )

def make_DOWN(cin, cout, num):
    tmp = []
    for i in range(num-1):
        tmp.append(nn.Conv2d(cin, cin, 4, 2, 1, bias=False))
        tmp.append(Norm2d(cin))
        tmp.append(nn.ReLU(inplace=True))
    tmp.append(nn.Conv2d(cin, cout, 4, 2, 1, bias=False))
    tmp.append(Norm2d(cout))
    return nn.Sequential(*tmp)

def make_stage(name, num, channels):
    tmp = dict()
    for i in range(num):
        for j in range(len(channels)):
            tmp["%s_%02d_%02d"%(name,i,j)] = make_RES4(channels[j])
        for j in range(len(channels)):
            for k in range(len(channels)):
                tmp_n = "%s_%02d_%02d_%02d"%(name,i,j,k)
                if j<k:
                    tmp[tmp_n] = make_DOWN(channels[j], channels[k], k-j)
                elif j==k:
                    tmp[tmp_n] = nn.Identity()
                elif j>k:
                    tmp[tmp_n] = nn.Sequential(
                        nn.Conv2d(channels[j], channels[k], 1, 1, 0, bias=False),
                        Norm2d(channels[k]),
                        nn.Upsample(scale_factor=1<<(j-k), mode='bilinear', align_corners=False)
                    )
    return nn.ModuleDict(tmp)

class HRNet(nn.Module):
    def __init__(self, i = 3, o = 256, c = [32, 64, 128, 256]):
        super(HRNet, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(i, 64, 4, 2, 1, bias=False),
            Norm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1, bias=False),
            Norm2d(64),
            nn.ReLU(inplace=True),
            BTNK(64, 64, 256),
            BTNK(256, 64, 256),
            BTNK(256, 64, 256),
            BTNK(256, 64, 256)
        )
        self.T0 = nn.Sequential(
            nn.Conv2d(256, c[0], 3, 1, 1, bias=False),
            Norm2d(c[0]),
            nn.ReLU(inplace=True)
        )
        self.T1 = nn.Sequential(
            nn.Conv2d(256, c[1], 4, 2, 1, bias=False),
            Norm2d(c[1]),
            nn.ReLU(inplace=True)
        )
        self.T2 = nn.Sequential(
            nn.Conv2d(c[1], c[2], 4, 2, 1, bias=False),
            Norm2d(c[2]),
            nn.ReLU(inplace=True)
        )
        self.T3 = nn.Sequential(
            nn.Conv2d(c[2], c[3], 4, 2, 1, bias=False),
            Norm2d(c[3]),
            nn.ReLU(inplace=True)
        )
        self.stages = make_stage("stage2", 1, c[0:2])
        self.stages.update(make_stage("stage3", 4, c[0:3]))
        self.stages.update(make_stage("stage4", 3, c[0:4]))
        self.ups = nn.ModuleList([
            nn.Identity(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        ])
        self.agg = nn.Sequential(
            nn.Conv2d(sum(c), o, 1, 1, 0, bias=False),
            Norm2d(o),
            nn.ReLU(inplace=True)
        )

    def forward_stage(self, x, name, m, n):
        for i in range(m):
            for j in range(n):
                x[j] = self.stages["%s_%02d_%02d"%(name,i,j)](x[j])
            y = []
            for k in range(n):
                tmp = []    
                for j in range(n):
                    tmp.append(self.stages["%s_%02d_%02d_%02d"%(name,i,j,k)](x[j]))
                y.append(F.relu(torch.stack(tmp, dim=0).sum(dim=0),inplace=True))
            x = y
        return x
    
    def forward(self, x):
        y = self.stage1(x)
        y = [self.T0(y), self.T1(y)]
        y = self.forward_stage(y, "stage2", 1, 2)
        y.append(self.T2(y[1]))
        y = self.forward_stage(y, "stage3", 4, 3)
        y.append(self.T3(y[2]))
        y = self.forward_stage(y, "stage4", 3, 4)
        for i in range(4):
            y[i] = self.ups[i](y[i])
        return self.agg(torch.cat(y, dim=1))

def make_head():
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(256, 128, 3, 1, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(128, 64, 3, 1, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 32, 3, 1, 1)
    )

def get_FOF(i=3+32, c=[48, 96, 192, 384]):
    return nn.Sequential(
        HRNet(i=i, o=256, c=c),
        make_head()
    )


if __name__ == "__main__":
    t = get_FOF().to("cuda:0")
    img = torch.randn((8,3+32,512,512),device="cuda:0")
    for i in range(10000):
        res = t(img)
        print(res.shape)
