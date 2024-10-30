import torch
import numpy as np
from torch import nn
from skimage.measure import marching_cubes

def toDevice(sample, device):
    if torch.is_tensor(sample):
        return sample.to(device, non_blocking=True)
    elif sample is None or type(sample) == str:
        return sample
    elif isinstance(sample, dict):
        return {k: toDevice(v, device) for k, v in sample.items()}
    else:
        return [toDevice(s, device) for s in sample]
    
def write(name, v, f):
    with open(name+".obj", "w") as mf:
        for i in v:
            mf.write("v %f %f %f\n" % (i[0], i[1], i[2]))
        for i in range(f.shape[0]):
            mf.write("f %d %d %d\n" % (f[i,0]+1, 
                                    f[i,1]+1,
                                    f[i,2]+1))
            
class Recon():
    def __init__(self, device, dim, res = 512) -> None:
        self.device = device
        self.res = res
        z = (torch.arange(res, dtype=torch.float32, device=device)+0.5)/res
        tmp = torch.arange(dim, dtype=torch.float32, device=device).view(1, dim)
        z =  tmp * z.view(res, 1) * np.pi
        self.z = torch.cos(z)
        self.z[:,0] = 0.5

    def decode(self, ceof):
        with torch.no_grad():
            res = torch.einsum("dc, chw -> dhw", self.z, ceof)
            v, f, _, _ = marching_cubes(res.cpu().numpy(), level = 0.5)
        v += 0.5
        v = v/(self.res/2) - 1
        v[:,1] *= -1
        vv = np.zeros_like(v)
        vv[:,0] = v[:,2]
        vv[:,1] = v[:,1]
        vv[:,2] = v[:,0]

        ff = np.zeros_like(f)
        ff[:,0] = f[:,0]
        ff[:,1] = f[:,2]
        ff[:,2] = f[:,1]
        return vv, ff

class Warp(nn.Module):
    def __init__(self, cc, device) -> None:
        super(Warp, self).__init__()
        self.device = device
        self.cc = cc
        y, x = torch.meshgrid(torch.arange(0, 512), torch.arange(0, 512))
        x = (x+0.5)/256-1
        y = 1-(y+0.5)/256
        self.yy = nn.Parameter(y[None][None].expand(cc,1,512,512),requires_grad=False).to(self.device)
        self.xx = nn.Parameter(x[None][None].expand(cc,1,512,512),requires_grad=False).to(self.device)
        self.num_tmp = nn.Parameter(torch.arange(1, 17)[None,:,None,None],requires_grad=False).to(self.device)
        
    def forward(self, features, srcs, tags, R, flag = False):
        cc = tags.shape[0]
        
        if cc != self.cc:
            xx = self.xx[0].expand(cc,1,512,512)
            yy = self.yy[0].expand(cc,1,512,512)
        else:
            xx = self.xx
            yy = self.yy
        points_3d = torch.cat([xx, yy, tags],dim=1)
        points_3d = torch.einsum("tdc, tchw -> tdhw",R, points_3d)
        
        coord = points_3d[:,:2,:,:]
        coord[:,1,:,:] *= -1
        coord = coord.permute([0,2,3,1])
        ww = torch.nn.functional.grid_sample(features, coord, padding_mode='zeros', align_corners=False)
        mask = tags>-1  # True 前景  False 背景
        ww = ww*mask
        zmap = points_3d[:,2:3,:,:]
        if not flag:
            zmap = zmap*mask
            ans = torch.cat([ww,
                            torch.cos(self.num_tmp*zmap*np.pi),
                            torch.sin(self.num_tmp*zmap*np.pi)
                            ],dim=1)
            return ans
        else:
            zmap_src = torch.nn.functional.grid_sample(srcs, coord, align_corners=False)
            masks = torch.abs((zmap_src - zmap)) <= 0.001 * (1+torch.minimum(zmap_src, zmap))
            return ww,masks*mask