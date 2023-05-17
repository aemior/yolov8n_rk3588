# Yolv8n RK3588 Demo :rocket:

This repository contains a demonstration of Yolv8n running on an RK3588 device.

## How to Use

### Configuring OpenCV Installation Directory

Edit CMakeList.txt to set your OpenCV installation directory:

```
set(OpenCV_DIR /root/lib/opencv454_install/lib/cmake/opencv4)
```

Replace the path with your own OpenCV installation directory.

### Building the Project

```shell
mkdir build
cd build
cmake ..
make
```

### Running the Demo

**Inference from Images**

To run the demo using default image:

```shell
rknn_yolov8_demo_picture
```

**Inference from Video**

Video inference is currently under development :construction:. Initial tests indicate performance is currently too slow.

```
rknn_yolov8_demo_video
```

## Notes

The tools directory contains scripts for model conversion which are intended to be run on a host machine. These include:

- modity_no_boxhead_transpose.py

  This script removes the bbox head and performs other operations on the official exported Yolv8n model.

- onnx2rknn_export.py

  This script converts the modified ONNX model (with the head removed) to an RKNN model.