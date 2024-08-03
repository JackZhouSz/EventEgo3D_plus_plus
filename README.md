# EventEgo3D++: 3D Human Motion Capture from a Head Mounted Event Camera 
<center>

Christen Millerdurai<sup>1,2</sup>, Hiroyasu Akada<sup>1</sup>, Jian Wang<sup>1</sup>, Diogo Luvizon<sup>1</sup>, 
Alain Pagani<sup>2</sup>, Didier Stricker<sup>2</sup>, Christian Theobalt<sup>1</sup>, Vladislav Golyanik<sup>1</sup>


<sup>1</sup> Max Planck Institute for Informatics, SIC  &nbsp; &nbsp; &nbsp; &nbsp; <sup>2</sup> DFKI Augmented Vision    

</center>

## Official PyTorch implementation

[Project page](https://eventego3d.mpi-inf.mpg.de/) | Paper(TBP) 

<p align="center">
<img src="images/teaser.gif" alt="EventEgo3D" height="172"  /></br>
</p>

### Abstract

Monocular egocentric 3D human motion capture remains a significant challenge, particularly under conditions of low lighting and fast movements, which are common in head-mounted device applications. Existing methods that rely on RGB cameras often fail under these conditions. To address these limitations, we introduce EventEgo3D++, the first approach that leverages a monocular event camera with a fisheye lens for 3D human motion capture. Event cameras excel in high-speed scenarios and varying illumination due to their high temporal resolution, providing reliable cues for accurate 3D human motion capture. EventEgo3D++ leverages the LNES representation of event streams to enable precise 3D reconstruction. We have also developed a mobile head-mounted device (HMD) prototype equipped with an event camera, capturing a comprehensive dataset that includes real event observations from both controlled studio environments and in-the-wild settings, in addition to a synthetic dataset. Additionally, to provide a more holistic dataset, we include allocentric RGB streams that offer different perspectives of the HMD wearer, along with their corresponding SMPL body model. Our experiments demonstrate that EventEgo3D++ achieves superior 3D accuracy and robustness compared to existing solutions, even in challenging conditions. Moreover, our method supports real-time 3D pose updates at a rate of 140Hz. This work is an extension of Millerdurai et al.'s EventEgo3D (CVPR 2024) and further advances the state of the art in egocentric 3D human motion capture.


### Advantages of Event Based Vision
High Speed Motion                      |  Low Light Performance          
:-------------------------:|:-------------------------:|
| <img src="images/fast_motion.gif" alt="High Speed Motion" width="350"/> | <img src="images/low_light.gif" alt="Low Light Performance" width="350"/> |

### Method

<p align="center">
<img src="images/architecture-min.png" alt="EventEgo3D" /></br>
</p>

## Usage
-----
- [EventEgo3D++: 3D Human Motion Capture from a Head Mounted Event Camera](#eventego3d-3d-human-motion-capture-from-a-head-mounted-event-camera)
  - [Official PyTorch implementation](#official-pytorch-implementation)
    - [Abstract](#abstract)
    - [Advantages of Event Based Vision](#advantages-of-event-based-vision)
    - [Method](#method)
  - [Usage](#usage)
    - [Installation](#installation)
      - [Dependencies](#dependencies)
      - [Pretrained Models](#pretrained-models)
    - [Datasets](#datasets)
    - [Training](#training)
    - [Evaluation](#evaluation)
      - [EE3D-S](#ee3d-s)
      - [EE3D-R](#ee3d-r)
      - [EE3D-W](#ee3d-w)
  - [Citation](#citation)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
------

### Installation

Clone the repository
```bash
git clone https://github.com/Chris10M/EventEgo3D_plus_plus.git
cd EventEgo3D_plus_plus
```

#### Dependencies
Create a conda enviroment from the file 
```bash
conda env create -f EventEgo3D.yml
```
Next, install  **[ocam_python](https://github.com/Chris10M/ocam_python.git)** using pip
```bash
pip3 install git+https://github.com/Chris10M/ocam_python.git
```


#### Pretrained Models 

The pretrained models for [EE3D-S](#ee3d-s), [EE3D-R](#ee3d-r) and [EE3D-W](#ee3d-w) can be downloaded from  

- [EE3D-S](https://eventego3d.mpi-inf.mpg.de/eventego3dplusplus/savedmodels/EE3D-S_pretrained_weights.pth) 
- [EE3D-R](https://eventego3d.mpi-inf.mpg.de/eventego3dplusplus/savedmodels/EE3D_R_finetuned_weights.pth)
- [EE3D-W](https://eventego3d.mpi-inf.mpg.de/eventego3dplusplus/savedmodels/EE3D_W_finetuned_weights.pth)

Please place the models in the following folder structure.

```bash
EventEgo3D_plus_plus
|
└── saved_models
         |
         └── EE3D-S_pretrained_weights.pth
         └── EE3D_R_finetuned_weights.pth
         └── EE3D_W_finetuned_weights.pth

```


### Datasets

The datasets can obtained by executing the files in [`dataset_scripts`](./dataset_scripts/). For detailed information, refer [here](./dataset_scripts/). 


### Training

For training, ensure [EE3D-S](./dataset_scripts#ee3d-s), [EE3D-R](./dataset_scripts#ee3d-r), [EE3D-W](./dataset_scripts#ee3d-w) and [EE3D[BG-AUG]](./dataset_scripts#ee3d-bg-aug) are present. 
The batch size and checkpoint path can be specified with the following environment variables, ```BATCH_SIZE``` and ```CHECKPOINT_PATH```.

```bash
python train.py 
```

### Evaluation

#### EE3D-S 
For evaluation, ensure [EE3D-S Test](./dataset_scripts#ee3d-s-test) is present. Please run, 

```bash
python evaluate_ee3d_s.py 
```

The provided [pretrained](#pretrained-models) checkpoint gives us an accuracy of,

| Arch | Head_MPJPE | Neck_MPJPE | Right_shoulder_MPJPE | Right_elbow_MPJPE | Right_wrist_MPJPE | Left_shoulder_MPJPE | Left_elbow_MPJPE | Left_wrist_MPJPE | Right_hip_MPJPE | Right_knee_MPJPE | Right_ankle_MPJPE | Right_foot_MPJPE | Left_hip_MPJPE | Left_knee_MPJPE | Left_ankle_MPJPE | Left_foot_MPJPE | MPJPE | Head_PAMPJPE | Neck_PAMPJPE | Right_shoulder_PAMPJPE | Right_elbow_PAMPJPE | Right_wrist_PAMPJPE | Left_shoulder_PAMPJPE | Left_elbow_PAMPJPE | Left_wrist_PAMPJPE | Right_hip_PAMPJPE | Right_knee_PAMPJPE | Right_ankle_PAMPJPE | Right_foot_PAMPJPE | Left_hip_PAMPJPE | Left_knee_PAMPJPE | Left_ankle_PAMPJPE | Left_foot_PAMPJPE | PAMPJPE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EgoHPE | 12.352 | 16.771 | 36.175 | 66.462 | 93.806 | 35.465 | 76.059 | 105.924 | 80.987 | 115.125 | 171.536 | 191.155 | 78.594 | 124.341 | 189.189 | 205.695 | 99.977 | 39.656 | 36.747 | 41.251 | 66.301 | 92.938 | 41.706 | 68.345 | 97.608 | 58.781 | 79.086 | 106.359 | 112.208 | 60.858 | 80.211 | 103.357 | 110.828 | 74.765 |


#### EE3D-R
For evaluation, ensure [EE3D-R](./dataset_scripts#ee3d-r) is present. Please run, 

```bash
python evaluate_ee3d_r.py 
```

The provided [pretrained](#pretrained-models) checkpoint gives us an accuracy of,

| Arch | walk_MPJPE | crouch_MPJPE | pushup_MPJPE | boxing_MPJPE | kick_MPJPE | dance_MPJPE | inter. with env_MPJPE | crawl_MPJPE | sports_MPJPE | jump_MPJPE | MPJPE | walk_PAMPJPE | crouch_PAMPJPE | pushup_PAMPJPE | boxing_PAMPJPE | kick_PAMPJPE | dance_PAMPJPE | inter. with env_PAMPJPE | crawl_PAMPJPE | sports_PAMPJPE | jump_PAMPJPE | PAMPJPE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EgoHPE | 67.692 | 142.700 | 89.626 | 135.720 | 96.960 | 84.902 | 93.841 | 108.606 | 98.586 | 96.665 | 101.530 | 49.499 | 96.272 | 70.492 | 110.577 | 79.146 | 69.245 | 66.646 | 79.597 | 74.972 | 76.766 | 77.321 |

#### EE3D-W
For evaluation, ensure [EE3D-W](./dataset_scripts#ee3d-w) is present. Please run, 

```bash
python evaluate_ee3d_w.py 
```

The provided [pretrained](#pretrained-models) checkpoint gives us an accuracy of,

| Arch | walk_MPJPE | crouch_MPJPE | pushup_MPJPE | boxing_MPJPE | kick_MPJPE | dance_MPJPE | inter. with env_MPJPE | crawl_MPJPE | sports_MPJPE | jump_MPJPE | MPJPE | walk_PAMPJPE | crouch_PAMPJPE | pushup_PAMPJPE | boxing_PAMPJPE | kick_PAMPJPE | dance_PAMPJPE | inter. with env_PAMPJPE | crawl_PAMPJPE | sports_PAMPJPE | jump_PAMPJPE | PAMPJPE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EgoHPE | 172.191 | 167.328 | 173.650 | 148.384 | 191.646 | 174.675 | 173.725 | 166.375 | 205.732 | 187.689 | 176.139 | 91.437 | 99.447 | 106.180 | 73.156 | 92.385 | 100.196 | 91.654 | 88.707 | 107.971 | 104.321 | 95.545 |


## Citation


## License

EventEgo3D++ is under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. The license also applies to the pre-trained models.

## Acknowledgements

The code is partially adapted from [here](https://github.com/microsoft/human-pose-estimation.pytorch). 

