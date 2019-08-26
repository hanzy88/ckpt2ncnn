
import tensorflow as tf

from tensorflow.python.platform import gfile

model = "./models/exp.pb"
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
for node in graph_def.node :
    if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in range(len(node.input)):
            print(node.input[index])
            if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
    elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: 
            del node.attr['use_locking']
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter('./logs/exp',graph)




