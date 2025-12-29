<div align="center"> 

## IR-ADS: Invariant Representation and Anomaly Separation for Robust Building Surface Defect Detection

Huadong Li, Xiao'ou Zhang, Haoyu Wang*, Xiaoyu Yi*, Xiang Li*, Mengmeng Liu, Fagen Li
</div>


## Abstract

Surface defect detection is crucial for ensuring the structural integrity of buildings. Traditional methods often struggle with distribution shifts caused by unknown materials or defect types. To address this, we propose IR-ADS, a method based on invariant representation learning and anomaly distribution separation. By introducing a depth-enhanced fusion strategy, IR-ADS aggregates geometric cues with RGB features, promoting view-invariant attributes. Additionally, an anomaly-separation mechanism, grounded in Schrödinger Bridge Theory, effectively segments unseen defect types. Extensive experiments demonstrate IR-ADS's robustness and lightweight model complexity, outperforming baselines by over 3.9% in mIoU.

![Network.png](Network.png)


## Environment Setup
- This repository is based on the Detrex framework.
- Python $\ge$ 3.7 and PyTorch $\ge$ 1.10 are required.  

- Install ```detectron2``` and ```detrex```
```
pip install -e detectron2
pip install -r requirements.txt
pip install -e .
```

- If you encounter any ```compilation error of cuda runtime```, you may try to use
```
export CUDA_HOME=<your_cuda_path>
```

- You may download images and annoations besides VOOC2007 from:
```shell
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```
- [Download DeepCrack Dataset](https://www.kaggle.com/datasets/cvvlearner/a-crack-dataset)
- [Download Khanh11k Dataset](https://github.com/khanhha/crack_segmentation/tree/master)
- [Download Masonry Dataset](https://www.kaggle.com/datasets/mexwell/crack-detection-in-images-of-bricks-and-masonry/data)
-  You may download images and annoations from: [Figshare](https://doi.org/10.6084/m9.figshare.30963293)
- and then organize the data as:
```
datasets/
├── DeepCrack 
│   ├── RGB
│   ├── HHA
│   └── Label
├── Khanh11k 
│   ├── Depth
│   ├── labels
│   ├── RGB
│   ├── test.txt
│   └── train.txt
├── Masonry 
│   ├── rgb
│   ├── ther
│   └── labels
├── VOOC2017/
│   │
│   ├── annotations/                  
│   │   ├── instances_train2017.json  
│   │   └── instances_val2017.json    
│   │
│   ├── train2017/                    
│   │   └── ...
│   │
│   └── val2017/                   
│       └── ...
│
├── style_train2017/
│   └── ...
│
├── train2017_depth_cmap/
    └── ...
```

## Training Step 1

Before training, please download [pre-trained Swin-Transformer](https://drive.google.com/drive/folders/1YqwIwt8e986zZC3omrTdhMrbiDEWwagw), and modify the relevant configuration settings accordingly:

```text
In semseg/models/backbones/swin.py (line 1108)
# For Swin-B
checkpoint_file = '/xxxx/xxxx/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'
# For Swin-L    
# checkpoint_file = '/xxxx/xxxx/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
```

To train model, please update the appropriate configuration file in `configs/` with appropriate dataset paths. Then run as follows:


```bash
python train_mm.py --cfg configs/nyu_rgbd.yaml
```

## Training Step 2
```
python projects/train_net.py \
    --config-file projects/vCLR_deformable_mask/configs/dino-resnet/deformable_train_voc_eval_nonvoc.py \
    --num-gpus N \
    dataloader.train.total_batch_size=8 \
    train.output_dir=<output_dir> \
    model.num_queries=2000 \ # similar performance when more than 1000
    train.amp.enabled=True \ # mixed precision training
    model.transformer.encoder.use_checkpoint=True \ # gradient checkpointing, save gpu memory but lower speed
    train.init_checkpoint=detectron2/dino_RN50_pretrain_d2_format.pkl \ # NOTE training from scratch is better for baseline model
```

```
python projects/train_net.py \
    --config-file projects/vCLR_deformable_mask/configs/dino-resnet/deformable_train_coco_eval_lvis.py \
    --num-gpus N \
    dataloader.train.total_batch_size=8 \
    train.output_dir=<output_dir> \
    model.num_queries=2000 \ # similar performance when more than 1000
    train.amp.enabled=True \ # mixed precision training
    model.transformer.encoder.use_checkpoint=True \ # gradient checkpointing, save gpu memory but lower speed
    train.init_checkpoint=detectron2/dino_RN50_pretrain_d2_format.pkl \ # NOTE training from scratch is better for baseline model
```

## Evaluation
To evaluate models, please download respective model weights ([**GoogleDrive**](https://drive.google.com/drive/folders/1YqwIwt8e986zZC3omrTdhMrbiDEWwagw)).
Then update the appropriate configuration file in `configs/` with appropriate dataset paths, and run:
```bash
python val_mm.py --cfg configs/nyu_rgbd.yaml
```

## Result
![Seg-1.png](Seg-1.png)

## License

All code, configuration files, and documentation in this repository are released under the  
**Creative Commons CC0 1.0 Universal (CC0 1.0)** license.

This means the authors dedicate the work to the public domain and waive all rights to the
maximum extent permitted by law.

This repository is intended for **research and reproducibility purposes**.

License details: https://creativecommons.org/publicdomain/zero/1.0/
