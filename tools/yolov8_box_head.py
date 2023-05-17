import numpy as np

def yolov8_box_head(input):
    layer_sizes = [80, 40, 20]
    layer_pointer = [0, 6400, 8000]
    strides = [8, 16, 32]
    for l in range(3):

        x = np.arange(0.5, layer_sizes[l], 1)
        y = np.arange(0.5, layer_sizes[l], 1)
        sx, sy = np.meshgrid(x, y)
        sx = sx.reshape(-1)
        sy = sy.reshape(-1)
        p_s, p_e = layer_pointer[l], layer_pointer[l] + layer_sizes[l]**2

        input[0,0,p_s:p_e] = sx - input[0,0,p_s:p_e] 
        input[0,1,p_s:p_e] = sy - input[0,1,p_s:p_e] 
        input[0,2,p_s:p_e] = sx + input[0,2,p_s:p_e] 
        input[0,3,p_s:p_e] = sy + input[0,3,p_s:p_e]

        c_x = (input[0,0,p_s:p_e] + input[0,2,p_s:p_e]) / 2
        c_y = (input[0,1,p_s:p_e] + input[0,3,p_s:p_e]) / 2
        w = input[0,2,p_s:p_e] - input[0,0,p_s:p_e]
        h = input[0,3,p_s:p_e] - input[0,1,p_s:p_e]

        input[0,0,p_s:p_e] = c_x
        input[0,1,p_s:p_e] = c_y
        input[0,2,p_s:p_e] = w
        input[0,3,p_s:p_e] = h

        input[0,:,p_s:p_e] *= strides[l]

    return input

if __name__ == "__main__":
    import pdb
    pdb.set_trace()
    arr = np.load("yolov8n_no_boxhead_result_0.npy")
    yolov8_box_head(arr)

