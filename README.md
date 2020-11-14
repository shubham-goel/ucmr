# Shape and Viewpoints without Keypoints

Shubham Goel, Angjoo Kanazawa, Jitendra Malik

University of California, Berkeley
In ECCV, 2020

[Project Page](https://shubham-goel.github.io/ucmr/)
![Teaser Image](https://shubham-goel.github.io/ucmr/resources/images/teaser.png)

### Requirements
- Python 3.7
- Pytorch 1.1.0
- Pymesh
- SoftRas
- NMR

### Installation
Please use [this Dockerfile](Dockerfile) to build your environment. For convenience, we provide a pre-built docker image at [shubhamgoel/birds](https://hub.docker.com/r/shubhamgoel/birds). If interested in a non-docker build, please follow [docs/installation.md](docs/installation.md)

### Training
Please see [docs/training.md](docs/training.md)

### Demo
1. From the `ucmr` directory, download the pretrained models:
```
wget https://people.eecs.berkeley.edu/~shubham-goel/projects/ucmr/cub_train_cam4_withcam.tar.gz && tar -vzxf cub_train_cam4_withcam.tar.gz
```
You should see `cachedir/snapshots/cam/e400_cub_train_cam4`

2. Run the demo:
```
python -m src.demo \
    --pred_pose \
    --pretrained_network_path=cachedir/snapshots/cam/e400_cub_train_cam4/pred_net_600.pth \
    --shape_path=cachedir/template_shape/bird_template.npy\
    --img_path demo_data/birdie1.png
```

### Evaluation
To evaluate camera poses errors on the entire test dataset, first download the CUB dataset and annotation files as instructed in [docs/training.md](docs/training.md). Then run
```bash
python -m src.experiments.benchmark \
        --pred_pose \
        --pretrained_network_path=cachedir/snapshots/cam/e400_cub_train_cam4/pred_net_600.pth \
        --shape_path=cachedir/template_shape/bird_template.npy \
        --nodataloader_computeMaskDt \
        --split=test
```

### Citation
If you use this code for your research, please consider citing:
```
@inProceedings{ucmrGoel20,
  title={Shape and Viewpoints without Keypoints},
  author = {Shubham Goel and
  Angjoo Kanazawa and
  and Jitendra Malik},
  booktitle={ECCV},
  year={2020}
}
```

### Acknowledgements
Parts of this code were borrowed from [CMR](https://github.com/akanazawa/cmr) and [CSM](https://github.com/nileshkulkarni/csm/).
