#include "rknn_net.h"
#include "yolov8_post.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>

int main () {

    rknn_net* yolov8 = rknn_net_create("../data/yolov8n_no_boxhead_transpose.rknn", false);
    cv::Mat img = cv::imread("../data/bus.jpg");
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img -= 128;
    printf("img width = %d, img height = %d\n", img.cols, img.rows);

    float** outputs;
    outputs = (float**)malloc(2 * sizeof(float*));
    outputs[0] = (float*)malloc(1*4*8400 * sizeof(float));
    outputs[1] = (float*)malloc(1*80*8400 * sizeof(float));

    rknn_net_inference(yolov8, (int8_t*)img.data, outputs);

    std::vector<DetectionResult> results = yolov8_post_process(outputs, 80, 1.0, 1.0, 0.4, 0.8, true);
    
    cv::Mat output_img = cv::imread("../data/bus.jpg");
    yolov8_draw_result(results, output_img, coco_classes, 80);
    cv::imwrite("./debug.png", output_img);

    rknn_net_destroy(yolov8);

    return 0;
}
