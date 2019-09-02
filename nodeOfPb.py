# -*- coding: utf-8 -*-

import argparse
import tensorflow as tf
import os

def create_graph(pb):
    with tf.gfile.FastGFile(pb, 'rb') as f:
        # 使用tf.GraphDef()定义一个空的Graph
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

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
        
        # Imports the graph from graph_def into the current default Graph.
        tf.import_graph_def(graph_def, name='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb_dir", '-p')
    parser.add_argument("--output", '-o')
    args = parser.parse_args()

    if not args.pb_dir:
        print("Please input as: python nodeOfPb.py -p 'model.pb' -o 'result.txt'")
        print("-p: the path of the saved pb files")
        print("-o: the path of saved txt file of pb node")
        exit(1)
    # 创建graph
    create_graph(args.pb_dir)
     
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    with open(args.output, 'w+') as f:
        for tensor_name in tensor_name_list:
            f.write(tensor_name+'\n')