import onnx
from onnx import helper

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

# Remove existing output
for output in list(graph.output):
    graph.output.remove(output)

# Create new Transpose nodes to reshape the output tensors
transpose_node0 = onnx.helper.make_node(
    'Transpose',
    inputs=['/model.22/dfl/Reshape_1_output_0'],
    outputs=['output0'],
    perm=[0, 2, 1]
)

transpose_node1 = onnx.helper.make_node(
    'Transpose',
    inputs=['/model.22/Sigmoid_output_0'],
    outputs=['output1'],
    perm=[0, 2, 1]
)

# Append the new Transpose nodes to the graph
graph.node.extend([transpose_node0, transpose_node1])

# Create new output Tensors with the specified shape and datatype (float32)
new_output0 = helper.make_tensor_value_info('output0', onnx.TensorProto.FLOAT, [1, 8400, 4])
new_output1 = helper.make_tensor_value_info('output1', onnx.TensorProto.FLOAT, [1, 8400, 80])

graph.output.extend([new_output0, new_output1])

# Cut the Box head
node_to_remove = ['/model.22/Shape', '/model.22/Gather', '/model.22/Add', '/model.22/Div', '/model.22/Mul', '/model.22/Mul_1', '/model.22/Slice', '/model.22/Slice_1', '/model.22/Sub', '/model.22/Add_1', '/model.22/Add_2', '/model.22/Sub_1', '/model.22/Div_1', '/model.22/Concat_4', '/model.22/Mul_2']

for i in range(3,13):
    node_to_remove.append("/model.22/Constant_%d" % i)

tmp_node_list = []
for r_node in node_to_remove:
    for node in graph.node:
        if node.name == r_node:
            tmp_node_list.append(node)
            break
for r_node in tmp_node_list:
    graph.node.remove(r_node)

# Save the modified model
onnx.save(model, '../data/yolov8n_no_boxhead_transpose.onnx')
