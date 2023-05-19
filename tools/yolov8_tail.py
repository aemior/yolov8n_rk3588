import numpy as np

def softmax(x):
    e_x = np.exp(x)
    s_e_x = np.sum(e_x, axis=-1).reshape(-1,4,1)
    return e_x/s_e_x

def box_tail(x):
    x = x.reshape((-1,4,16))
    x = softmax(x)
    return np.sum(x*np.arange(16).reshape(1,1,-1), axis=-1).reshape(-1,4)


def tail_process(network_outputs):
    a1 = network_outputs[1].transpose((0,2,3,1)).reshape(1,-1,64,1)
    a2 = network_outputs[3].transpose((0,2,3,1)).reshape(1,-1,64,1)
    a3 = network_outputs[5].transpose((0,2,3,1)).reshape(1,-1,64,1)
    bbox_20 = box_tail(network_outputs[1].transpose((0,2,3,1)).reshape(1,-1,64,1)).transpose().reshape((1,4,-1))
    bbox_40 = box_tail(network_outputs[3].transpose((0,2,3,1)).reshape(1,-1,64,1)).transpose().reshape((1,4,-1))
    bbox_80 = box_tail(network_outputs[5].transpose((0,2,3,1)).reshape(1,-1,64,1)).transpose().reshape((1,4,-1))
    bbox = np.concatenate((bbox_80, bbox_40, bbox_20), axis=2)

    confi_20 = network_outputs[0].reshape(1,80,-1)
    confi_40 = network_outputs[2].reshape(1,80,-1)
    confi_80 = network_outputs[4].reshape(1,80,-1)
    confi = np.concatenate((confi_80, confi_40, confi_20), axis=2)

    return [bbox, confi]

    

