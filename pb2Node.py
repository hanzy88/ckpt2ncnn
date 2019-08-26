# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 18:31:13 2018
1、model_dir为模型路径文件夹，model_name为模型名称（自定义非如alexnet等训练实际名称）
2、写入到模型路径下的result.txt文件内
@author: Mr_dogyang
"""
 
import tensorflow as tf
import os
 
model_dir = './models'
model_name = 'exp.pb'
 
# 读取并创建一个图graph来存放Google训练好的Inception_v3模型（函数）
def create_graph():
    with tf.gfile.FastGFile(os.path.join(
            model_dir, model_name), 'rb') as f:
        # 使用tf.GraphDef()定义一个空的Graph
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        """
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
        """
        # Imports the graph from graph_def into the current default Graph.
        tf.import_graph_def(graph_def, name='')
 
# 创建graph
create_graph()
 
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
result_file = os.path.join(model_dir, 'result.txt') 
with open(result_file, 'w+') as f:
    for tensor_name in tensor_name_list:
        f.write(tensor_name+'\n')