#include "yolov8_post.h"
#include <opencv2/opencv.hpp>
#include <cstring>
#include <iostream>
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

using namespace cv;

int l_size[3] = {80, 40, 20}; 
size_t l_base[3] = {0, 6400, 8000};
float striders[3] = {8, 16, 32};
int confi_idx[3] = {4,2,0};
int box_idx[3] = {5,3,1};


void get_confidence(float* input, int q_num, int group_size, int* idx, float* value) {
    int start = q_num * group_size;
    int end = start + group_size;
    float max_val = input[start];
    int max_idx = start;
    for (int i = start; i < end; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = i;
        }
    }
    *idx = max_idx - start;  // convert absolute index to index within the group
    *value = max_val;
}

float softmax_weightsum(float *input) {
    double base=0;
    float output=0;
    for (int i=0; i<16; ++i) {
        base += input[i];
    }
    for (int i=0; i<16; ++i) {
        output += i*(input[i]/base);
    }
    return output;
}
void printMaxElement(const cv::Mat& mat) {
    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    std::cout << "Max element: " << maxVal << std::endl;
}

void yolov8_tail_process(float** input, float** output, int class_num) {
    Timer t1;

    // Transpose and concatenate confidence data
    Mat confi_20(class_num, 400, CV_32F, input[0]);
    transpose(confi_20, confi_20);
    Mat confi_40(class_num, 1600, CV_32F, input[2]);
    transpose(confi_40, confi_40);
    Mat confi_80(class_num, 6400, CV_32F, input[4]);
    transpose(confi_80, confi_80);
    std::memcpy(output[1], confi_80.data, class_num*6400*sizeof(float));
    std::memcpy(output[1]+class_num*6400, confi_40.data, class_num*1600*sizeof(float));
    std::memcpy(output[1]+class_num*6400+class_num*1600, confi_20.data, class_num*400*sizeof(float));

    // Process bbox data
    float *ip, *op;
    Mat box_80_raw(64, 6400, CV_32F, input[5]); 
    transpose(box_80_raw, box_80_raw);
    exp(box_80_raw, box_80_raw);
    ip = (float*)box_80_raw.data;
    op = output[0];
    for (int i=0; i<6400; ++i) {
        for (int j=0; j<4; ++j) {
            op[i*4+j] = softmax_weightsum(ip+(i*64+j*16));
        }
    }
    Mat box_40_raw(64, 1600, CV_32F, input[3]); 
    transpose(box_40_raw, box_40_raw);
    exp(box_40_raw, box_40_raw);
    ip = (float*)box_40_raw.data;
    op = output[0]+6400*4;
    for (int i=0; i<1600; ++i) {
        for (int j=0; j<4; ++j) {
            op[i*4+j] = softmax_weightsum(ip+(i*64+j*16));
        }
    }
    Mat box_20_raw(64, 400, CV_32F, input[1]); 
    transpose(box_20_raw, box_20_raw);
    exp(box_20_raw, box_20_raw);
    ip = (float*)box_20_raw.data;
    op = output[0]+8000*4;
    for (int i=0; i<400; ++i) {
        for (int j=0; j<4; ++j) {
            op[i*4+j] = softmax_weightsum(ip+(i*64+j*16));
        }
    }

    std::cout << "tail_process:>" << t1.elapsed() << std::endl;
    return;
}

void get_max_confidence(float* data, size_t item_size, int pos, int class_num, float* value, int* class_id) {
    *value = 0;
    *class_id = 0;
    for (int i=0; i<class_num; ++i) {
        if (*value <= data[i * item_size + pos]) {
           *value = data[i * item_size + pos];
           *class_id = i;
        }
    }
}

float wsum_softmax(float *data, size_t item_size, int pos, int group) {
    double e_sum=0;
    float output=0;
    int base = group*16;
    for (int i=0; i<16; ++i) {
        e_sum += data[(base+i)*item_size+pos];
    }
    for (int i=0; i<16; ++i) {
        output += i*(data[(base+i)*item_size+pos]/e_sum);
    }
    return output;
}

void get_rawbbox(float *data, size_t item_size, int pos, int* x1, int* y1, int* x2, int* y2) {
}

std::vector<DetectionResult> yolov8_tail_post_process(float** input, int class_num, float scale_x, float scale_y, float score_threshold, float nms_threshold, bool debug_flag) {

    //Timer t1;

    std::vector<cv::Rect> boxes; // vector of bounding boxes
    std::vector<float> scores; // confidence scores for each box
    std::vector<int> indices; // indices of the raw detection result
    std::vector<int> cls; // vector of class IDs
    std::vector<DetectionResult> results; // result to return;

    float score, x1, y1, x2, y2, l_x, l_y, w, h;
    int class_id, cnt=0;
    size_t item_size, base_y;

    for (int l=0; l<3; ++l) {
        item_size = l_size[l]*l_size[l];
        Mat tmp_mat(64, item_size, CV_32F, input[box_idx[l]]);
        exp(tmp_mat, tmp_mat);
        for (int y=0; y<l_size[l]; ++y) {
            base_y = y*l_size[l];
            for (int x=0; x<l_size[l]; ++x) {
                get_max_confidence(input[confi_idx[l]], item_size, base_y+x, class_num, &score, &class_id);
                if (score < score_threshold) continue;
                x1 = wsum_softmax(input[box_idx[l]], item_size, base_y+x, 0);
                y1 = wsum_softmax(input[box_idx[l]], item_size, base_y+x, 1);
                x2 = wsum_softmax(input[box_idx[l]], item_size, base_y+x, 2);
                y2 = wsum_softmax(input[box_idx[l]], item_size, base_y+x, 3);

                l_x = x + 0.5 - x1;
                l_y = y + 0.5 - y1;
                w = (x1 + x2);
                h = (y1 + y2);
                boxes.push_back(cv::Rect(l_x*striders[l]*scale_x, l_y*striders[l]*scale_y, w*striders[l]*scale_x, h*striders[l]*scale_y));    
                scores.push_back(score);
                cls.push_back(class_id);
                indices.push_back(cnt++);
            }
        }
    }

    //Do NMS
    cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_threshold, indices);
    //std::cout << "tail_post_process:>" << t1.elapsed() << std::endl;
    if(debug_flag) {
        printf("========Detection Results========\n");
        printf("classID | score |     bbox\n");
        printf("---------------------------------\n");
    }
    for (int i=0; i<indices.size(); ++i) {
        int idx = indices[i];
        DetectionResult result = {scores[idx], cls[idx], boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height};
        results.push_back(result);
        if(debug_flag)
        printf("   %d    | %.2f  | %d %d %d %d\n", cls[idx], scores[idx], boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height);
    }
    if(debug_flag) printf("=================================\n");
    return results;
}

std::vector<DetectionResult> yolov8_post_process(float** input, int class_num, float scale_x, float scale_y, float score_threshold, float nms_threshold, bool debug_flag) {


    Timer t1;    
    std::vector<cv::Rect> boxes; // vector of bounding boxes
    std::vector<float> scores; // confidence scores for each box
    std::vector<int> indices; // indices of the raw detection result
    std::vector<int> cls; // vector of class IDs
    std::vector<DetectionResult> results; // result to return;

    //Some tmp var
    float score, l_x, l_y, w, h;
    int cl;
    size_t lbp, ybp, xbp, x1p, y1p, x2p, y2p;

    //Load detection results from buffers
    for (int l=0; l<3; ++l) {
        lbp = l_base[l];
        for (int y=0; y<l_size[l];++y) {
            ybp = lbp + y * l_size[l];
            for(int x=0; x<l_size[l]; ++x) {
                xbp = (ybp + x)*4;
                x1p = xbp;
                y1p = xbp + 1;
                x2p = xbp + 2;
                y2p = xbp + 3;
                
                l_x = x + 0.5 - input[0][x1p];
                l_y = y + 0.5 - input[0][y1p];
                w = (input[0][x2p] + input[0][x1p]);
                h = (input[0][y2p] + input[0][y1p]);

                boxes.push_back(cv::Rect(l_x*striders[l]*scale_x, l_y*striders[l]*scale_y, w*striders[l]*scale_x, h*striders[l]*scale_y));    
                get_confidence(input[1], ybp+x, class_num, &cl, &score);
                scores.push_back(score);
                cls.push_back(cl);
                indices.push_back(ybp+x);
            }
        }
    }

    //Do NMS
    cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_threshold, indices);
    std::cout << "post_process:>" << t1.elapsed() << std::endl;
    if(debug_flag) {
        printf("========Detection Results========\n");
        printf("classID | score |     bbox\n");
        printf("---------------------------------\n");
    }
    for (int i=0; i<indices.size(); ++i) {
        int idx = indices[i];
        DetectionResult result = {scores[idx], cls[idx], boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height};
        results.push_back(result);
        if(debug_flag)
        printf("   %d    | %.2f  | %d %d %d %d\n", cls[idx], scores[idx], boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height);
    }
    if(debug_flag) printf("=================================\n");
    return results;
}

cv::Scalar hsv_to_bgr(int h, int s, int v) {
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, s, v));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Scalar bgr_color = bgr.at<cv::Vec3b>(0,0);
    return bgr_color;
}

std::vector<cv::Scalar> get_colors(int nums_of_class) {
    std::vector<cv::Scalar> colors;
    for (int i=0; i<nums_of_class; ++i) {
        colors.push_back(hsv_to_bgr((int)((float)i/(float)nums_of_class*255),200,230));
    }
    return colors;
}

void yolov8_draw_result(std::vector<DetectionResult> results, cv::Mat &img, const std::string* class_names, int num_of_classes) {
    std::vector<cv::Scalar> colors = get_colors(num_of_classes);
    for (int i=0; i<results.size(); ++i) {
        cv::Rect bbox(results[i].x, results[i].y, results[i].w, results[i].h);
        cv::rectangle(img, bbox, colors[results[i].cls], 2);

        // Prepare the text label
        std::string label = class_names[results[i].cls] + ":" + std::to_string(results[i].score);
        
        // Choose a position for the label
        int baseLine;
        cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        // Draw filled box to put label text in it
        rectangle(img, cv::Rect(cv::Point(results[i].x, results[i].y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), colors[results[i].cls], cv::FILLED);

        // Put label text on the image
        cv::putText(img, label, cv::Point(results[i].x, results[i].y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0));
    }
}
