import math
import numpy as np
from numba import cuda
from scipy.io import loadmat
import cv2 

@cuda.jit(device=True)
def lock(mutex): 
    while cuda.atomic.compare_and_swap(mutex, 0, 1) != 0: pass 
    cuda.threadfence()

@cuda.jit(device=True) 
def unlock(mutex): 
    cuda.threadfence() 
    cuda.atomic.exch(mutex, 0, 0) 

@cuda.jit
def trans(v,vn,tmp_v,tmp_vn,r,res):
    vid = cuda.grid(1)
    if vid >= vn.shape[0]: return
    tmp_vn[vid,0] = vn[vid,0]*r[0,0] + vn[vid,1]*r[0,1] + vn[vid,2]*r[0,2]
    tmp_vn[vid,1] = vn[vid,0]*r[1,0] + vn[vid,1]*r[1,1] + vn[vid,2]*r[1,2]
    tmp_vn[vid,2] = vn[vid,0]*r[2,0] + vn[vid,1]*r[2,1] + vn[vid,2]*r[2,2]

    x = v[vid,0]*r[0,0] + v[vid,1]*r[0,1] + v[vid,2]*r[0,2]
    y = v[vid,0]*r[1,0] + v[vid,1]*r[1,1] + v[vid,2]*r[1,2]
    z = v[vid,0]*r[2,0] + v[vid,1]*r[2,1] + v[vid,2]*r[2,2]

    tmp_v[vid,0] = (1+x)*(res/2) - 0.5
    tmp_v[vid,1] = (1-y)*(res/2) - 0.5
    tmp_v[vid,2] = z

@cuda.jit
def raster(v, f, vn, depth, normal, mutex):
    fid = cuda.grid(1)
    lim = f.shape[0]
    res = depth.shape[0]
    if fid >= lim: return

    vid = f[fid]
    p1 = v[vid[0]]
    p2 = v[vid[1]]
    p3 = v[vid[2]]
    
    iMax = min(math.ceil(max(p1[0],p2[0],p3[0])), res) # x+1
    jMax = min(math.ceil(max(p1[1],p2[1],p3[1])), res) # x+1
    iMin = max(math.ceil(min(p1[0],p2[0],p3[0])), 0)
    jMin = max(math.ceil(min(p1[1],p2[1],p3[1])), 0)
    
    for j in range(jMin, jMax):
        for i in range(iMin, iMax):
            w3 = (p2[0]-p1[0])*(j-p1[1]) - (p2[1]-p1[1])*(i-p1[0])
            w1 = (p3[0]-p2[0])*(j-p2[1]) - (p3[1]-p2[1])*(i-p2[0])
            w2 = (p1[0]-p3[0])*(j-p3[1]) - (p1[1]-p3[1])*(i-p3[0])
            ss = w1+w2+w3
            if ss==0: continue
            if w1>=0 and w2>=0 and w3>=0: 
                d = (w1*p1[2]+w2*p2[2]+w3*p3[2])/ss
                lock(mutex[j,i])
                # add FOF here
                if depth[j,i,0] < d:
                    vn1 = vn[vid[0]]
                    vn2 = vn[vid[1]]
                    vn3 = vn[vid[2]]
                    n1 = (w1*vn1[0]+w2*vn2[0]+w3*vn3[0])/ss
                    n2 = (w1*vn1[1]+w2*vn2[1]+w3*vn3[1])/ss
                    n3 = (w1*vn1[2]+w2*vn2[2]+w3*vn3[2])/ss
                    nn = math.sqrt(n1*n1 + n2*n2 + n3*n3)
                    n1 = n1/nn
                    n2 = n2/nn
                    n3 = n3/nn
                    depth[j,i,0] = d
                    normal[j,i,0] = n1
                    normal[j,i,1] = n2
                    normal[j,i,2] = n3
                unlock(mutex[j,i]) 
            elif w1<=0 and w2<=0 and w3<=0:
                d = (w1*p1[2]+w2*p2[2]+w3*p3[2])/ss
                lock(mutex[j,i])
                # add FOF here
                if depth[j,i,0] < d:
                    vn1 = vn[vid[0]]
                    vn2 = vn[vid[1]]
                    vn3 = vn[vid[2]]
                    n1 = (w1*vn1[0]+w2*vn2[0]+w3*vn3[0])/ss
                    n2 = (w1*vn1[1]+w2*vn2[1]+w3*vn3[1])/ss
                    n3 = (w1*vn1[2]+w2*vn2[2]+w3*vn3[2])/ss
                    nn = math.sqrt(n1*n1 + n2*n2 + n3*n3)
                    n1 = n1/nn
                    n2 = n2/nn
                    n3 = n3/nn
                    depth[j,i,0] = d
                    normal[j,i,0] = n1
                    normal[j,i,1] = n2
                    normal[j,i,2] = n3
                unlock(mutex[j,i]) 

@cuda.jit
def get_vn_0(v, f, vn):
    fid = cuda.grid(1)
    if fid >= f.shape[0]: return
    vid = f[fid]
    p1 = v[vid[0]]
    p2 = v[vid[1]]
    p3 = v[vid[2]]
    x1 = (p2[0] - p1[0])*1024
    y1 = (p2[1] - p1[1])*1024
    z1 = (p2[2] - p1[2])*1024
    x2 = (p3[0] - p1[0])*1024
    y2 = (p3[1] - p1[1])*1024
    z2 = (p3[2] - p1[2])*1024
    nx = y1*z2-z1*y2
    ny = z1*x2-x1*z2
    nz = x1*y2-y1*x2
    nn = max(math.sqrt(nx*nx+ny*ny+nz*nz), 1e-8)
    nx = nx/nn
    ny = ny/nn
    nz = nz/nn
    for i in range(3):
        cuda.atomic.add(vn[vid[i]],0,nx)
        cuda.atomic.add(vn[vid[i]],1,ny)
        cuda.atomic.add(vn[vid[i]],2,nz)

@cuda.jit
def get_vn_1(vn):
    vid = cuda.grid(1)
    if vid >= vn.shape[0]: return
    n = vn[vid]
    tmp = max(math.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]), 1e-8)
    for i in range(3):
        n[i] /= tmp

def get_vn(v_cuda, f_cuda):
    vn_cuda = cuda.device_array_like(v_cuda)
    get_vn_0[(f_cuda.shape[0]+63)//64,64](v_cuda,f_cuda,vn_cuda)
    get_vn_1[(v_cuda.shape[0]+63)//64,64](vn_cuda)
    return vn_cuda

class Renderer():
    def __init__(self, v, f, res) -> None:
        self.res = res
        self.v_cuda = cuda.to_device(v)
        self.f_cuda = cuda.to_device(f)
        self.vn_cuda = get_vn(self.v_cuda, self.f_cuda)

        self.tmp_v_cuda = cuda.device_array_like(self.v_cuda)
        self.tmp_vn_cuda = cuda.device_array_like(self.vn_cuda)
        self.depth_cuda = cuda.device_array((res,res,1),dtype=np.float32)
        self.normal_cuda = cuda.device_array((res,res,3),dtype=np.float32)
        self.mutex_cuda = cuda.device_array((res,res,1),dtype=np.int32)
        self.mutex_cuda[:,:,:] = 0

    def render(self, R):
        R_cuda = cuda.to_device(R)
        self.depth_cuda[:,:,:] = -1
        self.normal_cuda[:,:,:] = 0
        trans[(self.v_cuda.shape[0]+63)//64,64](self.v_cuda, self.vn_cuda, self.tmp_v_cuda, self.tmp_vn_cuda, R_cuda, self.res)
        raster[(self.f_cuda.shape[0]+63)//64,64](self.tmp_v_cuda, self.f_cuda, self.tmp_vn_cuda, self.depth_cuda, self.normal_cuda, self.mutex_cuda)
        return self.normal_cuda.copy_to_host(), self.depth_cuda.copy_to_host()

def render_all(v, f, res=512):
    cuda.select_device(0)
    r = Renderer(v, f, res)
    normal, depth = r.render(np.eye(3))
    normal = normal[:,:,::-1]  
    return normal, depth