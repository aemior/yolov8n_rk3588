import onnx
from onnx import helper

import numpy as np

# Load the ONNX model
model = onnx.load('../data/yolov8n_full.onnx')

# Get the model's graph
graph = model.graph

# Find the Concat operation to remove
concat_node = None
for node in graph.node:
    if node.name == '/model.22/Concat_5':
        concat_node = node
        break

# If the Concat operation is found, remove it
if concat_node:
    graph.node.remove(concat_node)

# Find the nodes producing '/model.22/Mul_2_output_0' and '/model.22/Sigmoid_output_0' and change their outputs
for node in graph.node:
    node_outputs = list(node.output)
    if '/model.22/dfl/Reshape_1_output_0' in node_outputs:
        node_outputs[node_outputs.index('/model.22/dfl/Reshape_1_output_0')] = 'output0'
    elif '/model.22/Sigmoid_output_0' in node_outputs:
        node_outputs[node_outputs.index('/model.22/Sigmoid_output_0')] = 'output1'
    node.output[:] = node_outputs

# Remove existing output
for output in list(graph.output):
    graph.output.remove(output)

# Cut the Box head
node_to_remove = ['/model.22/Concat','/model.22/Reshape','/model.22/Concat_1','/model.22/Reshape_1','/model.22/Concat_2','/model.22/Reshape_2','/model.22/Concat_3','/model.22/Split','/model.22/dfl/Constant_1', '/model.22/dfl/Constant', '/model.22/Sigmoid', '/model.22/dfl/Reshape_1', '/model.22/dfl/conv/Conv', '/model.22/dfl/Transpose_1', '/model.22/dfl/Softmax', '/model.22/dfl/Transpose', '/model.22/dfl/Reshape', '/model.22/Shape', '/model.22/Gather', '/model.22/Add', '/model.22/Div', '/model.22/Mul', '/model.22/Mul_1', '/model.22/Slice', '/model.22/Slice_1', '/model.22/Sub', '/model.22/Add_1', '/model.22/Add_2', '/model.22/Sub_1', '/model.22/Div_1', '/model.22/Concat_4', '/model.22/Mul_2']

for i in range(1,13):
    node_to_remove.append("/model.22/Constant_%d" % i)

node_to_remove.append("/model.22/Constant")

tmp_node_list = []
for r_node in node_to_remove:
    for node in graph.node:
        if node.name == r_node:
            tmp_node_list.append(node)
            break
for r_node in tmp_node_list:
    graph.node.remove(r_node)

# Above is used to remove the tail

# Define new nodes
# 80x20x20
sigmoid_node_0 = helper.make_node(
    'Sigmoid',
    name='/tail/Sigmoid_0',
    inputs=['/model.22/cv3.2/cv3.2.2/Conv_output_0'],
    outputs=['/branch_20/confi'],
)

# 80x40x40
sigmoid_node_1 = helper.make_node(
    'Sigmoid',
    name='/tail/Sigmoid_1',
    inputs=['/model.22/cv3.1/cv3.1.2/Conv_output_0'],
    outputs=['/branch_40/confi'],
)

# 80x80x80
sigmoid_node_2 = helper.make_node(
    'Sigmoid',
    name='/tail/Sigmoid_2',
    inputs=['/model.22/cv3.0/cv3.0.2/Conv_output_0'],
    outputs=['/branch_80/confi'],
)
# Add new nodes to graph
graph.node.extend([sigmoid_node_0, sigmoid_node_1, sigmoid_node_2])
# Create new output Tensors with the specified shape and datatype (float32)
new_output0 = helper.make_tensor_value_info('/branch_20/confi', onnx.TensorProto.FLOAT, [1, 80, 20, 20])
new_output1 = helper.make_tensor_value_info('/model.22/cv2.2/cv2.2.2/Conv_output_0', onnx.TensorProto.FLOAT, [1, 64, 20, 20])

new_output2 = helper.make_tensor_value_info('/branch_40/confi', onnx.TensorProto.FLOAT, [1, 80, 40, 40])
new_output3 = helper.make_tensor_value_info('/model.22/cv2.1/cv2.1.2/Conv_output_0', onnx.TensorProto.FLOAT, [1, 64, 40, 40])

new_output4 = helper.make_tensor_value_info('/branch_80/confi', onnx.TensorProto.FLOAT, [1, 80, 80, 80])
new_output5 = helper.make_tensor_value_info('/model.22/cv2.0/cv2.0.2/Conv_output_0', onnx.TensorProto.FLOAT, [1, 64, 80, 80])

graph.output.extend([new_output0, new_output1, new_output2, new_output3, new_output4, new_output5])

# Check the model
result = onnx.shape_inference.infer_shapes(model)

# Save the modified model
onnx.save(result, 'yolov8n_no_tail.onnx')
