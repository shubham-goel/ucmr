## Downloading the CUB Data
1. Download CUB-200-2011 images.

```
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz && tar -xf CUB_200_2011.tgz
```

2. Download our annotation files and template meshes as follows. This should create `cachedir/cub` and `cachedir/template_shape`
```
wget https://people.eecs.berkeley.edu/~shubham-goel/projects/ucmr/cachedir.tar.gz && tar -vzxf cachedir.tar.gz
```

## Model Training
### Initialize camera multiplex
You may download precomputed NMR-initialized camera poses via
```
wget https://people.eecs.berkeley.edu/~shubham-goel/projects/ucmr/cub_init_campose8x5.tar.gz && tar -vzxf cub_init_campose8x5.tar.gz
```
This should create `cachedir/logs/cub_init_campose8x5`

*Note:* We used NMR to initialize our camera poses multiplex for all experiments because we computed these initializations once at the beginning of the project, and then continued using them. Softras could be used for this initialization but we didn't look into this very carefully.

To recompute NMR-initialized camera multiplex, run:
```bash
python -m src.experiments.cam_init --name=cub_init_campose8x5 --flagfile=configs/cub-init.cfg
```

### Train shape+texture
We train shape and texture in 3 steps:
```bash
# Step1: Train shape+texture using 40-camera multiplex for 21 epochs
python -m src.experiments.camOpt_shape --name=cub_train_cam8x5 --flagfile=configs/cub-train.cfg \
        --num_epochs=21 \
        --cameraPoseDict=cachedir/logs/cub_init_campose8x5/stats/campose_0.npz

# Step2: Prune camera poses to keep only top 4. This should create `.../raw_20_PRUNE4.npz`
python -m src.trim_cameras --input_file=cachedir/logs/cub_train_cam8x5/stats/raw_20.npz --topK=4 --noflipZ

# Step3: Train shape+texture using the pruned 4-camera multiplex for another 400 epochs
python -m src.experiments.camOpt_shape --name=cub_train_cam4 --flagfile=configs/cub-train.cfg \
        --num_epochs=400 --texture_flipCam --optimizeCamera_reloadCamsFromDict \
        --num_multipose_el=1 --num_multipose_az=4 \
        --cameraPoseDict=cachedir/logs/cub_train_cam8x5/stats/raw_20_PRUNE4.npz \
        --pretrained_network_path=cachedir/snapshots/cub_train_cam8x5/pred_net_20.pth
```

*Note:* While training shape+texture, you can simultaneously supervise a camera-pose-prediction head to predict the most-likely camera in the multiplex (which is being optimized) using the `--pred_pose --pred_pose_supervise` options. We haven't tested with this option extensively, but it could let you skip the feed-forward camera pose prediction in the next section.

### Train camera poses
After the shape+texture predictor finishes training and the camera-multiplex has been optimized, you may train a feed-forward camera pose predictor:
```bash
export EXP_NAME=cub_train_cam4
python -m src.experiments.pose_trainer \
        --name Cam/e400_"$EXP_NAME" \
        --flagfile=configs/cub-train.cfg \
        --pred_pose --camera_loss_wt=2 \
        --nooptimizeCameraCont \
        --batch_size=128 --num_epochs=800 --learning_rate=0.0001 \
        --nodataloader_computeMaskDt \
        --use_cameraPoseDict_as_gt --nocameraPoseDict_dataloader_isCamPose \
        --cameraPoseDict_dataloader=cachedir/logs/"$EXP_NAME"/stats/raw_399.npz \
        --pretrained_network_path=cachedir/snapshots/"$EXP_NAME"/pred_net_400.pth
```


## Other Categories
You may download annotation files, precomputed camera-multiplex initializations and pretrained models for other categories via
```
wget https://people.eecs.berkeley.edu/~shubham-goel/projects/ucmr/cachedir-others.tar.gz && tar -vzxf cachedir-others.tar.gz
```
### Pascal Car
```bash
# Camera-initialization
python -m src.experiments.cam_init --name=car_init_campose8x1 --flagfile=configs/car-init.cfg

# Train shape+tex
python -m src.experiments.camOpt_shape --name=car_train_cam8 --flagfile=configs/car-train.cfg \
        --num_epochs=500 \
        --cameraPoseDict=cachedir/logs/car_init_campose8/stats/campose_0.npz

# Train camera
export EXP_NAME=car_train_cam8
python -m src.experiments.pose_trainer \
        --name Cam/e400_"$EXP_NAME" \
        --flagfile=configs/cub-train.cfg \
        --pred_pose --camera_loss_wt=2 \
        --nooptimizeCameraCont \
        --batch_size=128 --num_epochs=800 --learning_rate=0.0001 \
        --nodataloader_computeMaskDt \
        --use_cameraPoseDict_as_gt --nocameraPoseDict_dataloader_isCamPose \
        --cameraPoseDict_dataloader=cachedir/logs/"$EXP_NAME"/stats/raw_399.npz \
        --cameraPoseDict_dataloader_mergewith=cachedir/logs/"$EXP_NAME"/stats/raw_398.npz \
        --pretrained_network_path=cachedir/snapshots/"$EXP_NAME"/pred_net_400.pth

# Benchmark/Evaluate
python -m src.experiments.benchmark \
        --pred_pose \
        --pretrained_network_path=cachedir/snapshots/cam/e400_car_train_cam8/pred_net_600.pth \
        --shape_path=cachedir/template_shape/car_template.npy\
        --nodataloader_computeMaskDt --batch_size=1\
        --split=val --dataset=p3d --p3d_class=car
```

To benchmark 3DIoU, run evaluation script above with the `--save_mats` flag. This saves all shape predictions as mat files to `cachedir/evaluation/p3d_test/`. To evaluate these shapes, download (from [here](https://people.eecs.berkeley.edu/~shubham-goel/projects/ucmr/p3d_eval.zip)) and extract preprocessed pascal3d shapes to `cachedir/p3d_eval`. Also, point to the correct PASCAL3D+_release1.1 dataset path in `src/p3d_iou/driver.m`.
```bash
# Save predictions to mat files
python -m src.experiments.benchmark \
        --pred_pose \
        --pretrained_network_path=cachedir/snapshots/cam/e400_car_train_cam8/pred_net_600.pth \
        --shape_path=cachedir/template_shape/car_template.npy\
        --nodataloader_computeMaskDt --batch_size=1\
        --split=val --dataset=p3d --p3d_class=car\
        --save_mats 

# Compute 3DIoU using matlab script.
cd src/p3d_iou/
matlab -nodisplay < driver.m
```
