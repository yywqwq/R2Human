# R²Human: Real-Time 3D Human Appearance Rendering from a Single Image (ISMAR 2024)

**News**

* `30/10/2024` The code is released. But it's not complete. I'm still updating it. 

### [Project Page](https://cic.tju.edu.cn/faculty/likun/projects/R2Human) | [Paper](https://arxiv.org/pdf/2312.05826) 

> [R²Human: Real-Time 3D Human Appearance Rendering from a Single Image](https://arxiv.org/pdf/2312.05826)  
> Yuanwang Yang, Qiao Feng, Yu-Kun Lai, Kun Li
> ISMAR 2024

We plan to release the training code of R2Human in this repository as soon as possible. Any discussions or questions would be welcome!


## Dependencies

To run this code, the following packages need to be installed.

```
conda create -n r2human python=3.8
conda activate r2human

# visit https://pytorch.org/
# install latest pytorch
# for example:
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install tqdm
pip install opencv-python
pip install scikit-image
pip install numba==0.58.1
```


## Pretrained model

You can get pretrained model [here](https://drive.google.com/drive/folders/1zeKXJo3bQdQGqkgqSMpmbO7D7-A3F6ds?usp=sharing). Put them in `./ckpt`. They should be organized as shown below:
```
R2Human
├─lib
└─ckpt
    ├─model.pth
    └─netF.pth
```

## Testing
The input images should be `.png`s with 512*512 resolution in RGB format. And we need the foreground mask and the estimated smpl under orthogonal projection (can be estimated by [4dhuman](https://github.com/shubham-goel/4D-Humans) or [pymaf](https://github.com/HongwenZhang/PyMAF)). They should be organized as shown below:
```
R2Human
├─lib
└─input
    └─0000
        ├─img.png
        ├─mask.png
        └─smpl.npz
```
You can test it by running the following command, and the results are saved in `./output/`:
```
python test.py
```

## Citation
If you find this work useful for your research, please use the following BibTeX entry. 


```
@inproceedings{R2Human,
  author = {Yuanwang Yang and Qiao Feng and Yu-Kun Lai and Kun Li},
  title = {R²Human: Real-Time 3D Human Appearance Rendering from a Single Image},
  booktitle = {2024 IEEE International Symposium on Mixed and Augmented Reality (ISMAR)},
  year={2024}
}
```



