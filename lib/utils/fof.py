import numpy as np
from numba import jit
from numba import njit
from numba.np.extensions import cross2d

@jit(nopython=True)
def get_mpi(v, f, res):
    v[:,1] *= -1 # y axis down
    v[:,:2] = (v[:,:2]+1)*(res/2) - 0.5
    mpi = []
    for fid in f:
        pts = v[fid]
        iMax = int(np.ceil(np.max(pts[:,0])))
        iMax = min(res, iMax)
        iMin = int(np.ceil(np.min(pts[:,0])))
        iMin = max(0, iMin)
        jMax = int(np.ceil(np.max(pts[:,1])))
        jMax = min(res, jMax)
        jMin = int(np.ceil(np.min(pts[:,1])))
        jMin = max(0, jMin)
        for i in range(iMin, iMax):
            for j in range(jMin, jMax):
                p = np.array([i,j])
                w2 = cross2d(pts[1,:2] - pts[0,:2], p - pts[0,:2])
                w0 = cross2d(pts[2,:2] - pts[1,:2], p - pts[1,:2])
                w1 = cross2d(pts[0,:2] - pts[2,:2], p - pts[2,:2])
                ss = w0+w1+w2
                if ss==0:
                    continue
                elif ss>0:
                    if w0>=0 and w1>=0 and w2>=0:
                        mpi.append((j*res+i, (w0*pts[0,2]+w1*pts[1,2]+w2*pts[2,2])/ss, 0))
                elif ss<0:
                    if w0<=0 and w1<=0 and w2<=0:
                        mpi.append((j*res+i, (w0*pts[0,2]+w1*pts[1,2]+w2*pts[2,2])/ss, 1))
    mpi = sorted(mpi)
    pos = []
    ind = []
    val = []
    pre = 0
    cnt = 0
    last = -1
    while pre < len(mpi):
        while pre<len(mpi) and mpi[pre][2]==1:
            pre+=1
        if pre>=len(mpi):
            break
        nxt = pre+1

        flag = False
        while True:
            if nxt >= len(mpi) or mpi[nxt][0] != mpi[pre][0]:
                flag=True
                break
            if mpi[nxt][2]==1 and (nxt+1 >= len(mpi) or mpi[nxt+1][0] != mpi[pre][0] or mpi[nxt+1][2]==0):
                flag=False
                break
            nxt += 1

        if flag:
            pre = nxt
        else:
            if mpi[pre][0]!=last:
                pos.append(mpi[pre][0])
                ind.append(cnt)
                last = mpi[pre][0]
            val.append(mpi[pre][1])
            val.append(mpi[nxt][1])
            cnt+=2
            pre = nxt+1
    pos = np.array(pos, dtype=np.uint32)
    val = np.array(val, dtype=np.float32)
    ind = np.array(ind, dtype=np.uint32)
    return pos, ind, val

@njit
def get_fof(pos, ind, val, res = 512):
    fof = np.zeros((res*res, 32), dtype=np.float32)
    ind = np.append(ind, len(val))
    for i in range(len(pos)):
        pid = pos[i]
        for j in range(ind[i], ind[i+1], 2):
            t2 = val[j+1]+1
            t1 = val[j]+1
            fof[pid, 0] += t2-t1
            for k in range(1, 32, 1):
                fof[pid, k] += np.sin(t2*0.5*k*np.pi)-np.sin(t1*0.5*k*np.pi)
        for k in range(1, 32, 1):
            fof[pid, k] /= 0.5*k*np.pi
    return fof.reshape((res,res,32))

def get_ceof(v, f, res=512):
    pos, ind, val = get_mpi(v, f, res)
    ceof = get_fof(pos, ind, val, res)
    return ceof