#include "rknn_net.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

const char* get_type(rknn_tensor_type type) {
    switch (type) {
        case RKNN_TENSOR_FLOAT32:
            return "float32";
        case RKNN_TENSOR_FLOAT16:
            return "float16";
        case RKNN_TENSOR_INT8:
            return "int8";
        case RKNN_TENSOR_UINT8:
            return "uint8";
        case RKNN_TENSOR_INT16:
            return "int16";
        case RKNN_TENSOR_UINT16:
            return "uint16";
        case RKNN_TENSOR_INT32:
            return "int32";
        case RKNN_TENSOR_UINT32:
            return "uint32";
        case RKNN_TENSOR_INT64:
            return "int64";
        case RKNN_TENSOR_BOOL:
            return "bool";
        default:
            return "unknown";
    }
}


rknn_net* rknn_net_create(const char* model_path, int debug_flag) {

    FILE* fp = fopen(model_path, "rb");
    if(fp == NULL) {
        printf("Failed to open file %s\n", model_path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long model_size = ftell(fp);
    rewind(fp);

    uint8_t* model = (uint8_t*)malloc(model_size);
    if(model == NULL) {
        printf("Failed to malloc memory for model\n");
        fclose(fp);
        return NULL;
    }

    size_t read_count = fread(model, 1, model_size, fp);
    if(read_count != model_size) {
        printf("Failed to read model\n");
        free(model);
        fclose(fp);
        return NULL;
    }

    if (debug_flag) {
        printf("read model from %s\n", model_path);
        printf("model size = %ld bytes\n", model_size);
    }

    fclose(fp);

    //rknn_context* ctx = NULL;
    rknn_context* ctx = (rknn_context*)malloc(sizeof(rknn_context));
    int ret = rknn_init(ctx, model, model_size, 0, NULL);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        free(model);
        free(ctx);
        return NULL;
    }

    rknn_sdk_version version;
    ret = rknn_query(*ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if(ret < 0) {
        printf("rknn_query SDK version error, ret=%d\n", ret);
        free(model);
        free(ctx);
        return NULL;
    }

    if (debug_flag) {
        printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);
    }

    rknn_input_output_num io_num;
    ret = rknn_query(*ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if(ret < 0) {
        printf("rknn_query IN_OUT_NUM error, ret=%d\n", ret);
        free(model);
        free(ctx);
        return NULL;
    }

    if (debug_flag) {
        printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    }

    rknn_tensor_attr* input_attrs = (rknn_tensor_attr*)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    rknn_tensor_attr* output_attrs = (rknn_tensor_attr*)calloc(io_num.n_output, sizeof(rknn_tensor_attr));

    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(*ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_query INPUT_ATTR error, ret=%d\n", ret);
            free(model);
            free(ctx);
            free(input_attrs);
            free(output_attrs);
            return NULL;
        }
    }

    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(*ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_query OUTPUT_ATTR error, ret=%d\n", ret);
            free(model);
            free(ctx);
            free(input_attrs);
            free(output_attrs);
            return NULL;
        }
    }

    int channel = 3;
    int width  = 0;
    int height  = 0;

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        if (debug_flag) {
            printf("model is NCHW input fmt\n");
        }
        channel = input_attrs[0].dims[1];
        height  = input_attrs[0].dims[2];
        width  = input_attrs[0].dims[3];
    } else {
        if (debug_flag) {
            printf("model is NHWC input fmt\n");
        }
        height         = input_attrs[0].dims[1];
        width  = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    if (debug_flag) {
        printf("model input height=%d, width=%d, channel=%d, type:%s\n", height, width, channel, get_type(input_attrs[0].type));
        for (int i=0; i<io_num.n_output; ++i) {
            printf("model output %d:", i);
            for (int j=0; j<output_attrs[i].n_dims; ++j) {
                printf("%d ", output_attrs[i].dims[j]);
            }
            printf("type:%s\n", get_type(output_attrs[0].type));
        }
    }

    // allocate and initialize rknn_net structure
    rknn_net* net = (rknn_net*)malloc(sizeof(rknn_net));
    net->ctx = ctx;
    net->debug_flag = debug_flag;
    net->model = model;
    net->model_size = model_size;
    net->input_attrs = input_attrs;
    net->output_attrs = output_attrs;
    net->num_input = io_num.n_input;
    net->num_output = io_num.n_output;
    net->input_height = height;
    net->input_width = width;
    net->input_channel = channel;

    return net;
}

int rknn_net_inference(rknn_net* net, int8_t* input, float** output) {
    rknn_input inputs[net->num_input];
    memset(inputs, 0, sizeof(inputs));

    inputs[0].index = 0;
    inputs[0].buf = (void*)input;
    inputs[0].size = net->input_attrs[0].n_elems * sizeof(int8_t);
    inputs[0].pass_through = true;
    inputs[0].type = RKNN_TENSOR_INT8;
    inputs[0].fmt = net->input_attrs[0].fmt;//RKNN_TENSOR_NCHW;

    int ret = rknn_inputs_set(*(net->ctx), 1, inputs);
    if(ret < 0) {
        printf("rknn_inputs_set fail! ret=%d\n", ret);
        return 1;
    } else if (net->debug_flag) {
        printf("rknn_inputs_set done.\n");
    }

    ret = rknn_run(*(net->ctx), NULL);
    if(ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return 1;
    } else if (net->debug_flag) {
        printf("rknn_run done.\n");
    }

    rknn_output outputs[2];
    memset(outputs, 0, sizeof(outputs));

    for (int i=0; i<2; ++i) {
        outputs[i].want_float = true;
        outputs[i].is_prealloc = true;
        outputs[i].buf = output[i];
        outputs[i].index = i;
        outputs[i].size = net->output_attrs[0].n_elems * sizeof(float);
    }

    ret = rknn_outputs_get(*(net->ctx), 2, outputs, NULL);
    if(ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return 1;
    } else if (net->debug_flag) {
        printf("rknn_outputs_get done.\n");
    }

    for (int i=0; i<2; ++i) {
        memcpy(output[i], outputs[i].buf, outputs[i].size);
    }
    rknn_outputs_release(*(net->ctx), 2, outputs);

    return 0;
}

void rknn_net_destroy(rknn_net* net) {
    if (net == NULL) {
        return;
    }

    // Release rknn context
    if (net->ctx != NULL) {
        rknn_destroy(*(net->ctx));
        net->ctx = NULL;
    }

    // Release model data
    if (net->model != NULL) {
        free(net->model);
        net->model = NULL;
    }

    // Release input tensor attributes
    if (net->input_attrs != NULL) {
        free(net->input_attrs);
        net->input_attrs = NULL;
    }

    // Release output tensor attributes
    if (net->output_attrs != NULL) {
        free(net->output_attrs);
        net->output_attrs = NULL;
    }

    // Finally, free the rknn_net struct itself
    free(net);
}