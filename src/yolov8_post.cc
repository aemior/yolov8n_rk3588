#include "yolov8_post.h"

int l_size[3] = {80, 40, 20}; 
size_t l_base[3] = {0, 6400, 8000};
float striders[3] = {8, 16, 32};


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

std::vector<DetectionResult> yolov8_post_process(float** input, int class_num, float scale_x, float scale_y, float score_threshold, float nms_threshold, bool debug_flag) {

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
    if(debug_flag) printf("================\n");
    for (int i=0; i<indices.size(); ++i) {
        int idx = indices[i];
        DetectionResult result = {scores[idx], cls[idx], boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height};
        results.push_back(result);
        if(debug_flag)
        printf(">%d:%f| %d %d %d %d|\n", cls[idx], scores[idx], boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height);
    }
    if(debug_flag) printf("================\n");
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
