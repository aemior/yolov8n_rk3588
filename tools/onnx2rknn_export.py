import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
import yolov8_box_head
from yolov8_tail import tail_process

from rknn.api import RKNN

ONNX_MODEL = '../data/yolov8n_no_boxhead_transpose.onnx'
RKNN_MODEL = '../data/yolov8n_no_boxhead_transpose.rknn'
IMG_PATH = '../data/bus.jpg'
DATASET = '../data/dataset.txt'

QUANTIZE_ON = True

TAIL_PROC = True

OBJ_THRESH = 0.5
NMS_THRESH = 0.45
IMG_SIZE = 640

CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

def post_process_yolov8(network_output, img_input, save_path=None):
    global OBJ_THRESH, NMS_THRESH, CLASSES, HEAD_PROC

    # Parameters
    confThreshold = OBJ_THRESH
    nmsThreshold = NMS_THRESH

    # Lists to store detected bounding boxes, confidences and class IDs
    boxes = []
    confidences = []
    classIDs = []

    # Process the raw output
    for detection in np.transpose(network_output, (2,1,0)):
    # Extract the bounding box coordinates and class confidences
        box = detection[0:4]
        class_confidences = detection[4:]
        # Get the class ID and its confidence
        classID = np.argmax(class_confidences)
        confidence = class_confidences[classID]

        # Filter out weak predictions
        if confidence > confThreshold:
            # Update our list of bounding boxes, confidences and class IDs
            boxes.append([int(val) for val in box])
            confidences.append(float(confidence))
            classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    if len(indices) > 0:
    # Loop over the indices we are keeping
        for i in indices.flatten():
            # Extract the bounding box coordinates
            (x, y, w, h) = [int(val) for val in boxes[i]]
            x = x - w//2
            y = y - h//2
            text = "{}: {:.4f}".format(CLASSES[classIDs[i]], confidences[i])
            print(text, "|", "%d,%d,%d,%d" % (x,y,w,h))

            if save_path != None:
                # Draw a bounding box rectangle and label on the image
                color = [int(c) for c in np.random.randint(0, 255, size=(3,))]
                cv2.rectangle(img_input, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img_input, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if save_path != None:
        cv2.imwrite(save_path, img_input)





if __name__ == '__main__':


    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]], target_platform="rk3588")
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Build model.')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')
    """

    # Hybird quantization step 1
    print('--> Hybird quantization step one')
    ret = rknn.hybrid_quantization_step1(dataset=DATASET)
    if ret != 0:
        print('Step one failed!')
        exit(ret)
    print('done')

    # Call hybrid_quantization_step2 to generate hybrid quantized RKNN model
    print('--> Hybird quantization step two')
    ret = rknn.hybrid_quantization_step2(
            model_input="./"+ONNX_MODEL.split('.')[0]+".model",
            data_input="./"+ONNX_MODEL.split('.')[0]+".data",
            model_quantization_cfg="./"+ONNX_MODEL.split('.')[0]+".quantization.cfg"
            )
    if ret != 0:
        print('Step two failed!')
        exit(ret)
    print('done')
    """

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (IMG_SIZE, IMG_SIZE), swapRB=True, crop=False)

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[blob], data_format="nchw", inputs_pass_through=[1])
    print('done')

    if (TAIL_PROC):
        outputs = tail_process(outputs)
        outputs[0] = yolov8_box_head.yolov8_box_head(outputs[0])

    merge_output = np.concatenate(outputs, axis=1)

    print('--> Post Process')
    post_process_yolov8(merge_output, img, "./"+ONNX_MODEL.split('.')[0]+"_result.png")
    print('done')

    rknn.release()
