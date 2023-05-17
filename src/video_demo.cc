#include "rknn_net.h"
#include "yolov8_post.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>

class Timer {
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
};

int main () {
    //std::cout << cv::getBuildInformation() << std::endl;
    // Default parameters
    std::string input_path = "/root/data/test_video.mp4";
    std::string model_path = "../data/yolov8n_no_boxhead_transpose.rknn";
    std::string output_path = "ffmpeg -f rawvideo -pix_fmt bgr24 -s 640x640 -i - -an -f rtsp rtsp://localhost:8848/test_video";
    float score_threshold = 0.4;
    float nms_threshold = 0.5;
    bool debug_flag = false;
    cv::Mat frame, output_img;
    //Temp var
    float scale_x, scale_y;

    // Allocate data buffer to save the raw result from Neural net
    float** outputs;
    outputs = (float**)malloc(2 * sizeof(float*));
    outputs[0] = (float*)malloc(1*4*8400 * sizeof(float));
    outputs[1] = (float*)malloc(1*80*8400 * sizeof(float));

    // Load the model
    rknn_net* yolov8 = rknn_net_create(model_path.c_str(), false);

    //Open the video
    cv::VideoCapture cap("/root/data/test_video.mp4");
    if(!cap.isOpened()) {
        printf("Video Open Faild\n");
        return -1;
    }  // check if we succeeded


    //Setup output
    //FILE* ffmpeg = popen(output_path.c_str(), "w");
    //if (!ffmpeg) {
    //    fprintf(stderr, "Could not open pipe to ffmpeg\n");
    //    return 1;
    //}

    Timer timer;
    int cnt=0;
    while(true)
    {
        if (++cnt > 10)
            break;
        cap >> frame; // get a new frame from camera/video or read image
        if(frame.empty())
            break;

        // Resize the image to 640x640
        cv::resize(frame, frame, cv::Size(640, 640));
        output_img = frame.clone();

        // Convert img properly
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        frame -= 128;


        // Do inference
        timer.reset();
        rknn_net_inference(yolov8, (int8_t*)frame.data, outputs);
        std::cout << "Elapsed time: " << timer.elapsed() << " seconds" << std::endl;

        // Post process to get the detection result
        std::vector<DetectionResult> results = yolov8_post_process(outputs, 80, 1, 1, 0.4, 0.8, debug_flag);

        // Draw the detection result
        yolov8_draw_result(results, output_img, coco_classes, 80);

        //fwrite(output_img.data, sizeof(char), output_img.total()*output_img.channels(), ffmpeg);
    }

    // Release the resource
    rknn_net_destroy(yolov8);
    //pclose(ffmpeg);
    return 0;
}
