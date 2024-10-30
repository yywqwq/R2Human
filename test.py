import os,sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
import imageio
from lib.network.HRNet import get_HRNet
from lib.network.FN import get_FOF
from lib.network.UNet import get_UNet
from lib.utils.utils import Warp,Recon,toDevice,write
from lib.utils.render import render_all
from lib.utils.fof import get_ceof

def get_r(vid, frames):
    y = np.pi*(vid)/frames*2
    sy = np.sin(y)
    cy = np.cos(y)
    r = np.array([  [ cy, 0.0,  sy],
                    [0.0, 1.0, 0.0],
                    [-sy, 0.0,  cy],] , dtype=np.float32)
    return r

def test(args, netF, netE, netD, data, ee, warp, frames, device):
    netF.eval()
    netE.eval()
    netD.eval()
    with torch.no_grad():
        data = toDevice(data,device)
        # 1. get fof & normal & depth
        fof = netF(torch.cat([data["img"],data["smpl"]], dim=1)).to(device)
        v,f = ee.decode(fof[0]*data["mask"][0])
        normal, depth = render_all(v,f,res=512)
        normal = torch.from_numpy(normal.astype(np.float32)).permute(2,0,1)[None].to(device)
        depth = torch.from_numpy(depth.astype(np.float32)).permute(2,0,1)[None].to(device)
        ceof = torch.zeros_like(fof).to(device)

        # 2. run
        f_src = netE.forward(torch.cat([data["img"]*2-1,ceof,normal],dim=1))
        os.makedirs("%s/%s"%(args["output"],data["name"]), exist_ok=True)
        imglist = []
        for i in tqdm(range(frames)):
            r = get_r(i, frames)
            normal_tag,depth_tag = render_all(np.matmul(v, r.T),f,res=512)
            normal_tag = torch.from_numpy(normal_tag.astype(np.float32)).permute(2,0,1)[None].to(device)
            depth_tag = torch.from_numpy(depth_tag.astype(np.float32)).permute(2,0,1)[None].to(device)
            f_tag = warp(f_src,depth,depth_tag,torch.from_numpy(np.linalg.inv(r))[None].to(device)).to(device)
            img = netD.forward(torch.cat([f_tag,normal_tag],dim=1))
            if i > 0:
                ww,mask = warp(pre["img"], pre["depth"], depth_tag, torch.from_numpy(np.linalg.inv(get_r(1, frames)))[None].to(device), flag=True)
                ww = ww.to(device)
                mask = mask.to(device)
                y = 1-0.5/(frames/2)*abs(i-frames/2)
                img[mask.expand(img.shape)]=(1-y)*img[mask.expand(img.shape)]+y*ww[mask.expand(img.shape)]
            pre = {
                "img"   : img,
                "depth" : depth_tag
            }
            img = (img*(depth_tag>-1))[0].permute(1,2,0).cpu().numpy()*127.5+127.5
            imglist.append(cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2BGR))
            cv2.imwrite("%s/%s/%03d.png"%(args["output"],data["name"],i),img)
        imageio.mimsave("%s/%s.gif"%(args["output"],data["name"]),imglist,'GIF',duration=0.08)


def main(args):
    device = torch.device(args["device"])
    print("using device:", device)

    print('Resuming from ', args["ckpt"])
    # models
    state_dict = torch.load(args["ckpt"], map_location="cpu")
    netE = get_HRNet().to(device)
    netE.load_state_dict(state_dict["netE"])
    netD = get_UNet().to(device)
    netD.load_state_dict(state_dict["netD"])
    del state_dict
    
    state_dict = torch.load("./ckpt/netF.pth", map_location="cpu")
    netF = get_FOF().to(device)
    netF.load_state_dict(state_dict["model"])
    del state_dict

    ee = Recon(device, 32, 512)
    fr = 1
    warp = Warp(1,device)

    # loop
    namelist = sorted(os.listdir(args["input"]))
    for i in range(len(namelist)):
        print("Running",namelist[i])
        img = os.path.join(args["input"],namelist[i],"img.png")
        img = torch.from_numpy(cv2.imread(img)).permute(2,0,1)[None]
        mask = os.path.join(args["input"],namelist[i],"mask.png")
        mask = torch.from_numpy(cv2.imread(mask,0))[None]!=0
        img = img*mask/255
        smpl = np.load(os.path.join(args["input"],namelist[i],"mesh.npz"))
        fof_smpl = get_ceof(smpl["v"],smpl["f"])
        fof_smpl = torch.from_numpy(fof_smpl).permute(2,0,1)[None]
        data = {
            "name"   : namelist[i],
            "img"    : img,
            "mask"   : mask,
            "smpl"   : fof_smpl,
        }
        test(args, netF, netE, netD, data, ee, warp, args["frames"], device)

if __name__ == "__main__":
    #--------------------cfg here--------------------
    args = {
        "input"  : "./input/",
        "output" : "./output/",
        "ckpt"   : "./ckpt/model.pth",
        "frames" : 64,
        "device" : "cuda:0"
    }
    main(args)