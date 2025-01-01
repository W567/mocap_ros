# mocap_ros 

ROS1 wrapper package for motion capture with [hand_object_detector](https://github.com/ddshan/hand_object_detector.git), [frankmocap](https://github.com/facebookresearch/frankmocap.git), [HaMeR](https://github.com/geopavlakos/hamer.git) and [4D-Humans](https://github.com/shubham-goel/4D-Humans.git).

![Alt text](asset/mocap_example.gif)

## Setup

### Prerequisite
This package is build upon
- ROS1 (Noetic)
- `torch` and `cuda`
- (Optional) docker and nvidia-container-toolkit (for environment safety)

### Build package

#### on your workspace
It is better to use docker environment cause it needs specific cuda version and build environment.
```bash
mkdir -p ~/ros/catkin_ws/src && cd ~/ros/catkin_ws/src
git clone https://github.com/ojh6404/mocap_ros.git
cd ~/ros/catkin_ws/src/mocap_ros
./prepare.sh # install torch and build python submodules
cd ~/ros/catkin_ws && catkin b
```

#### using docker (Recommended)
Otherwise, you can build this package on docker environment.
```bash
git clone https://github.com/ojh6404/mocap_ros.git
cd mocap_ros && catkin bt # to build message
docker build -t mocap_ros .
# docker build --no-cache -t mocap_ros .
```

## Usage
### 1. run directly
```bash
roslaunch mocap_ros hand_mocap_original.launch \
    input_image:=/kinect_head/rgb/image_rect_color \
    device:=cuda:0 \
    with_mocap:=true
```
### 2. using docker
You can run on docker by
```bash
./run_docker -host pr1040 -launch hand_mocap_original.launch \
    input_image:=/kinect_head/rgb/image_rect_color \
    device:=cuda:0 \
    with_mocap:=true
```

```
./run_docker -host localhost -launch hand_mocap.launch input_image:=/camera/color/image_rect_color input_depth:=/camera/aligned_depth_to_color/image_raw camera_info:=/camera/color/camera_info
```

where
- `-host` : hostname like `pr1040` or `localhost`
- `-launch` : launch file name to run

launch args below.
- `input_image` : input image topic
- `device` : which device to use. `cpu` or `cuda`. default is `cuda:0`.
- `hand_threshold` : hand detection threshold. default is `0.9`.
- `object_threshold` : object detection threshold. default is `0.9`.
- `with_handmocap` : use mocap or not. if you need faster detection and don't need mocap, then set `false`. default is `true`.

### Output topic
- `~hand_detections` : `MocapDetectionArray`. array of hand detection results. please refer to `msg`
- `~debug_image` : `Image`. image for visualization.
- `tf` : `tf`. (Optional) publish transforms of key points.

### TODO
add rostest and docker build test.


### Realsense:

```bash
./run_docker -host localhost -launch hand_mocap_original.launch input_image:=/camera/color/image_rect_color input_depth:=/camera/aligned_depth_to_color/image_raw camera_info:=/camera/color/camera_info track_3d:=false detector_model:=mediapipe_hand visualize:=true with_mocap:=true decompress:=false threshold:=0.95 margin:=20
```

### astra camera:

```bash
./run_docker -host localhost -launch hand_mocap_original.launch input_image:=/camera/color/image_raw input_depth:=/camera/depth/image_raw camera_info:=/camera/color/camera_info track_3d:=false detector_model:=mediapipe_hand visualize:=true with_mocap:=true decompress:=false threshold:=0.95 margin:=20
```


### new launch file:

Use camera_type to choose which camera to use. (astra or realsense or kinect)
Corresponding topics are set in the launch file based on the camera type.
```bash
/run_docker -host localhost -launch hand_mocap.launch camera_type:=astra detector_model:=mediapipe_hand visualize:=false with_mocap:=true threshold:=0.95 margin:=20
```