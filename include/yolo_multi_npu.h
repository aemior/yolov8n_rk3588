#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "yolov8_post.h"
#include "rknn_net.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>

template <typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable cond_empty;
    std::condition_variable cond_full;
    size_t max_size;

public:
    ThreadSafeQueue(size_t max_size) : max_size(max_size) {}

    void enqueue(T t) {
        std::unique_lock<std::mutex> lock(mutex);
        cond_full.wait(lock, [&](){ return queue.size() < max_size; });
        queue.push(t);
        cond_empty.notify_one();
    }

    T dequeue() {
        std::unique_lock<std::mutex> lock(mutex);
        cond_empty.wait(lock, [&](){ return !queue.empty(); });
        T val = queue.front();
        queue.pop();
        cond_full.notify_one();
        return val;
    }
};


#define BUFFER_SIZE 150
#define ELEMENT_SIZE 3*640*640

typedef struct {
    int8_t* data[BUFFER_SIZE];
    int start; // Index of the oldest element
    int end;   // Index at which to write new element
    pthread_mutex_t lock;
    pthread_cond_t notEmpty; // Signaled when the buffer is not empty
    pthread_cond_t notFull;  // Signaled when the buffer is not full
    bool stop;  // Flag to stop the thread
} CircularBuffer;

typedef struct {
    cv::VideoCapture* cam;
    CircularBuffer* output;    
} video_read_arg;

typedef struct {
    CircularBuffer* input;    
    CircularBuffer* output;    
    int* target_npu;
    std::string* model_path;
} yolov8_npu_arg;

void initialize(CircularBuffer *buffer) {
    buffer->start = 0;
    buffer->end = 0;
    for (int i=0; i< BUFFER_SIZE;++i) {
        buffer->data[i] = (int8_t*)malloc(ELEMENT_SIZE * sizeof(int8_t));
    }
    pthread_mutex_init(&buffer->lock, NULL);
    pthread_cond_init(&buffer->notEmpty, NULL);
    pthread_cond_init(&buffer->notFull, NULL);
    buffer->stop = false;
}

void enqueue(CircularBuffer *buffer, int8_t* input) {
    pthread_mutex_lock(&buffer->lock);
    while ((buffer->end + 1) % BUFFER_SIZE == buffer->start) { // Buffer is full
        pthread_cond_wait(&buffer->notFull, &buffer->lock);   // Wait until not full
        if(buffer->stop) {
            pthread_mutex_unlock(&buffer->lock);
            buffer->end = 1;
            buffer->start = 0;
            return;
        }
    }
    if(buffer->stop) {
        pthread_mutex_unlock(&buffer->lock);
        return;
    }
    memcpy(buffer->data[buffer->end], input, ELEMENT_SIZE * sizeof(int8_t));
    buffer->end = (buffer->end + 1) % BUFFER_SIZE;
    pthread_cond_signal(&buffer->notEmpty); // Signal that the buffer is not empty
    pthread_mutex_unlock(&buffer->lock);
}

void dequeue(CircularBuffer *buffer, int8_t* output) {
    pthread_mutex_lock(&buffer->lock);
    while (buffer->end == buffer->start) { // Buffer is empty
        pthread_cond_wait(&buffer->notEmpty, &buffer->lock); // Wait until not empty
        if(buffer->stop) {
            pthread_mutex_unlock(&buffer->lock);
            buffer->end = 1;
            buffer->start = 0;
            return;
        }
    }
    if(buffer->stop) {
        pthread_mutex_unlock(&buffer->lock);
        return;
    }
    memcpy(output, buffer->data[buffer->start], ELEMENT_SIZE * sizeof(int8_t));
    buffer->start = (buffer->start + 1) % BUFFER_SIZE;
    pthread_cond_signal(&buffer->notFull); // Signal that the buffer is not full
    pthread_mutex_unlock(&buffer->lock);
}

void release(CircularBuffer *buffer) {
    for (int i=0; i<BUFFER_SIZE; ++i) {
        free(buffer->data[i]);
    }
    free(buffer);
}

void* video_read_thread(void* args) {
    video_read_arg* video_args = (video_read_arg*)args;
    cv::VideoCapture* cam = video_args->cam;
    CircularBuffer* output_buffer = video_args->output;
    cv::Mat frame;
    fprintf(stdout, "Video Thread start!\n");
    while (true)
    {
        *cam >> frame;
        if(frame.empty())
            break;
        // Resize the image to 640x640
        cv::resize(frame, frame, cv::Size(640, 640));
        enqueue(output_buffer, (int8_t*)frame.data);
        if(output_buffer->stop) break;
    }
    fprintf(stdout, "Video Thread stop!\n");
    output_buffer->stop = true;
    pthread_cond_signal(&output_buffer->notEmpty); // Signal that the buffer is not empty
    return NULL;
}

void* yolov8_npu_thread(void* args) {
    yolov8_npu_arg* yolo_args = (yolov8_npu_arg*)args;
    std::string* model_path = yolo_args->model_path;
    int* target_npu = yolo_args->target_npu;
    CircularBuffer* input_buffer = yolo_args->input;
    CircularBuffer* output_buffer = yolo_args->output;

    rknn_net* yolov8 = rknn_net_create((*model_path).c_str(), *target_npu, false);

    int8_t* input;
    input = (int8_t*)malloc(3*640*640*sizeof(int8_t));

    float* outputs[6];
    outputs[0] = (float*)malloc(1*80*400 * sizeof(float));
    outputs[1] = (float*)malloc(1*64*400 * sizeof(float));
    outputs[2] = (float*)malloc(1*80*1600 * sizeof(float));
    outputs[3] = (float*)malloc(1*64*1600 * sizeof(float));
    outputs[4] = (float*)malloc(1*80*6400 * sizeof(float));
    outputs[5] = (float*)malloc(1*64*6400 * sizeof(float));

    fprintf(stdout, "NPU Thread start, NPU:%d\n", *yolo_args->target_npu);
    while (true)
    {
        dequeue(input_buffer, input);
        if(input_buffer->stop) break;
        rknn_net_inference(yolov8, input, outputs);
        std::vector<DetectionResult> results = yolov8_tail_post_process(outputs, 80, 1.0, 1.0, 0.3, 0.4, false);
        cv::Mat output_img(640, 640, CV_8UC3, (uint8_t*)input);
        yolov8_draw_result(results, output_img, coco_classes, 80);
        enqueue(output_buffer, (int8_t*)output_img.data);
    }
    rknn_net_destroy(yolov8);
    for (int i=0; i<6; ++i) {
        free(outputs[i]);
    }
    output_buffer->stop = true;
    pthread_cond_signal(&output_buffer->notEmpty); // Signal that the buffer is not empty
    fprintf(stdout,"NPU Thread stop, NPU:%d\n", *yolo_args->target_npu);
    return NULL;
}