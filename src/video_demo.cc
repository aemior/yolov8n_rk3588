#include "rknn_net.h"
#include "yolov8_post.h"
#include "yolo_multi_npu.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <pthread.h>
#include <signal.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <chrono>

void preciseSleep(double seconds)
{
    struct timespec ts;
    ts.tv_sec = (time_t)seconds; // Truncate the decimal part for the 'seconds' component
    ts.tv_nsec = (long)((seconds - ts.tv_sec) * 1e+9); // Subtract the 'seconds' component from 'seconds', leaving the fractional part

    nanosleep(&ts, NULL);
}

volatile sig_atomic_t flag = 0;
CircularBuffer input_buffer, output_buffer;

void handler(int sig) {
    flag = 1;
    output_buffer.stop = true;
    pthread_cond_signal(&output_buffer.notEmpty);
    pthread_cond_signal(&output_buffer.notFull);
    printf("\nCtrl+C Press, Wait Thread stop.\n");
    input_buffer.stop = true;
    pthread_cond_signal(&input_buffer.notEmpty);
    pthread_cond_signal(&input_buffer.notFull);
    sleep(2);
    printf("STOPED!\n");
}


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
    std::string model_path = "../data/yolov8n_no_tail.rknn";
    std::string output_path = "ffmpeg -loglevel quiet -f rawvideo -pix_fmt bgr24 -s 640x640 -r 28 -i - -an -b:v 20M -f rtsp rtsp://localhost:8848/test_video";
    float score_threshold = 0.4;
    float nms_threshold = 0.5;
    bool debug_flag = false;
    cv::Mat frame, output_img;
    //Temp var
    float scale_x, scale_y;
    signal(SIGINT, handler);

    cv::VideoCapture cap(input_path);
    if(!cap.isOpened()) {
        printf("Video Open Faild\n");
        return -1;
    }  // check if we succeeded


    //Setup output
    FILE* ffmpeg = popen(output_path.c_str(), "w");
    if (!ffmpeg) {
        fprintf(stderr, "Could not open pipe to ffmpeg\n");
        return 1;
    }
    int8_t* result_img = (int8_t*)malloc(3*640*640*sizeof(int8_t));

    //Setup FIFO
    initialize(&input_buffer);
    initialize(&output_buffer);

    pthread_t threads[4];
    int npu_0=0, npu_1=1, npu_2=2;
    yolov8_npu_arg npu_arg_0, npu_arg_1, npu_arg_2;
    npu_arg_0.input = &input_buffer;
    npu_arg_0.output = &output_buffer;
    npu_arg_0.model_path = &model_path;
    npu_arg_0.target_npu = &npu_0;
    pthread_create(&threads[0], NULL, yolov8_npu_thread, &npu_arg_0);
    npu_arg_1.input = &input_buffer;
    npu_arg_1.output = &output_buffer;
    npu_arg_1.model_path = &model_path;
    npu_arg_1.target_npu = &npu_1;
    pthread_create(&threads[1], NULL, yolov8_npu_thread, &npu_arg_1);
    npu_arg_2.input = &input_buffer;
    npu_arg_2.output = &output_buffer;
    npu_arg_2.model_path = &model_path;
    npu_arg_2.target_npu = &npu_2;
    pthread_create(&threads[2], NULL, yolov8_npu_thread, &npu_arg_2);

    video_read_arg video_arg;
    video_arg.cam = &cap;
    video_arg.output = &input_buffer;
    pthread_create(&threads[3], NULL, video_read_thread, &video_arg);

    Timer timer;
    size_t cnt=0;
    fprintf(stdout, "Wait for buffering!\n");
    sleep(5);
    fprintf(stdout, "Main process start!\n");
    while(!flag)
    {

        timer.reset();
        dequeue(&output_buffer, result_img);
        if (output_buffer.stop) break;
        if (timer.elapsed() < 0.033) preciseSleep(0.033-timer.elapsed());
        fwrite(result_img, sizeof(char), 3*640*640, ffmpeg);
        fprintf(stdout, "\rFPS: %f, total frame:%d    ", 1/timer.elapsed(), ++cnt);
    }

    fprintf(stdout, "\nMain process stop!\n");
    input_buffer.stop = true;
    pthread_cond_signal(&input_buffer.notEmpty);
    pthread_cond_signal(&input_buffer.notFull);

    // Release the resource
    release(&input_buffer);
    release(&output_buffer);
    free(result_img);
    pclose(ffmpeg);
    return 0;
}
