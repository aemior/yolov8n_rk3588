#include <getopt.h>
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

void print_help() {
    printf("Usage: your_program [OPTIONS]\n");
    printf("Options:\n");
    printf("\t-i, --input\t\tPath to the input image\n");
    printf("\t-m, --model\t\tPath to the model file\n");
    printf("\t-o, --output\t\tPath to the output image\n");
    printf("\t-d, --debug\t\tDebug flag (0 or 1)\n");
    printf("\t-s, --score_threshold\tScore threshold for filtering boxes\n");
    printf("\t-n, --nms_threshold\tNMS threshold\n");
    printf("\t-h, --help\t\tPrint this help message\n");
}

int main (int argc, char* argv[]) {
    // Define options
    const struct option long_options[] = {
        {"input",  required_argument, 0, 'i'},
        {"model",  required_argument, 0, 'm'},
        {"output", required_argument, 0, 'o'},
        {"debug",  required_argument, 0, 'd'},
        {"help",   no_argument,       0, 'h'},
        {"score_threshold", required_argument, 0, 's'},
        {"nms_threshold",   required_argument, 0, 'n'},
        {0, 0, 0, 0}
    };

    // Default parameters
    std::string input_path = "../data/bus.jpg";
    std::string model_path = "../data/yolov8n_no_boxhead_transpose.rknn";
    std::string output_path = "./debug.png";
    float score_threshold = 0.4;
    float nms_threshold = 0.5;
    bool debug_flag = true;

    //Temp var
    float scale_x, scale_y;

    // Parse options
    int opt = 0;
    while ((opt = getopt_long(argc, argv, "i:m:o:d:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'i':
                input_path = optarg;
                break;
            case 'm':
                model_path = optarg;
                break;
            case 'o':
                output_path = optarg;
                break;
            case 's':
                score_threshold = atof(optarg);
                break;
            case 'n':
                nms_threshold = atof(optarg);
                break;
            case 'd':
                debug_flag = atoi(optarg);
                break;
            case 'h':
            default:
                print_help();
                return 0;
        }
    }

    // Load the model
    rknn_net* yolov8 = rknn_net_create(model_path.c_str(), false);
    
    // Read image
    cv::Mat img = cv::imread(input_path);
    scale_x = ((float)img.cols) / 640;
    scale_y = ((float)img.rows) / 640;
    cv::Mat output_img = img.clone();
    
    // Resize the image to 640x640
    cv::resize(img, img, cv::Size(640, 640));

    // Convert img properly
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img -= 128;

    // Allocate data buffer to save the raw result from Neural net
    float** outputs;
    outputs = (float**)malloc(2 * sizeof(float*));
    outputs[0] = (float*)malloc(1*4*8400 * sizeof(float));
    outputs[1] = (float*)malloc(1*80*8400 * sizeof(float));

    // Do inference
    rknn_net_inference(yolov8, (int8_t*)img.data, outputs);

    // Post process to get the detection result
    std::vector<DetectionResult> results = yolov8_post_process(outputs, 80, scale_x, scale_y, 0.4, 0.8, debug_flag);

    // Draw the detection result
    yolov8_draw_result(results, output_img, coco_classes, 80);

    // Write the image which have detection result
    cv::imwrite(output_path, output_img);

    // Release the resource
    rknn_net_destroy(yolov8);
    

    return 0;
}

