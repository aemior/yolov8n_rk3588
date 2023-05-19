#ifndef YOLOV8_POST_H
#define YOLOV8_POST_H
#include <stddef.h> // or <stdlib.h> for size_t
#include <stdint.h> // for uint8_t
#include <stdlib.h>
#include <vector>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

typedef struct {
    float score;
    int cls;
    int x;
    int y;
    int w;
    int h;
} DetectionResult;

const std::string coco_classes[] = {"person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop ", "mouse   ", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush"};

/** @brief Performs tail process for yolov8(no tail, just like dfl).

* @param input input[0] is the buffer of confidence of layer_20x20 with shape (1xclass_numx20x20),
*              intpu[1] is the buffer of bbox of layer_20x20 with shape (1x64x20x20),
*              input[2] is the buffer of confidence of layer_40x40 with shape (1xclass_numx40x40),
*              intpu[3] is the buffer of bbox of layer_40x40 with shape (1x64x40x40),
*              input[4] is the buffer of confidence of layer_80x80 with shape (1xclass_numx80x80),
*              intpu[5] is the buffer of bbox of layer_80x80 with shape (1x64x80x80),
* @param output output[0] is the buffer of dlf result with shape (1x8400x4),
*              output[1] is the buffer of the confidence with shape (1x8400xclass_num).
* @param class_num the total number of classes;
*/
void yolov8_tail_process(float** input, float** output, int class_num);

/** @brief Performs post process for yolov8(no bbox head).

* @param input input[0] is the buffer of dlf result with shape (1x8400x4),
*              intpu[1] is the buffer of the confidence with shape (1x8400xclass_num).
* @param class_num the total number of classes;
* @param scale_x the scale of width from origin img compare to 640 equal to origin_img.width/640.
* @param scale_y the scale of height from origin img compare to 640 equal to origin_img.height/640.
* @param score_threshold a threshold used to filter boxes by score.
* @param nms_threshold a threshold used in non maximum suppression.
* @param debug_flat if print the debug imformation.
*/
std::vector<DetectionResult> yolov8_post_process(float** input, int class_num, float scale_x, float scale_y, float score_threshold, float nms_threshold, bool debug_flag);
std::vector<DetectionResult> yolov8_tail_post_process(float** input, int class_num, float scale_x, float scale_y, float score_threshold, float nms_threshold, bool debug_flag);

/** @brief Draw the detection result to image.

* @param result detection results.
* @param img image to draw.
* @param class_namess the name of each class.
* @param num_of_classes the total number of classes.
*/
void yolov8_draw_result(std::vector<DetectionResult> results, cv::Mat &img, const std::string* class_names, int num_of_classes);



#endif // YOLOV8_POST_H