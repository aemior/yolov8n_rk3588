#ifndef RKNN_NET_H
#define RKNN_NET_H

#include "rknn_api.h"
#include <stddef.h> // or <stdlib.h> for size_t
#include <stdint.h> // for uint8_t
#include <stdlib.h>


// Define the struct
typedef struct rknn_net {
    rknn_context* ctx;
    int debug_flag;
    uint8_t* model;
    size_t model_size;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int num_input;
    int num_output;
    int input_height;
    int input_width;
    int input_channel;
} rknn_net;


// Function prototypes
rknn_net* rknn_net_create(const char* model_path, int debug_flag);
int rknn_net_inference(rknn_net* net, int8_t* input, float** output);
void rknn_net_destroy(rknn_net* net);
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

#endif // RKNN_NET_H
