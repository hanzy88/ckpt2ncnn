import os
import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile

def createLogs(pb, log_path):
    graph = tf.get_default_graph()
    graph_def = graph.as_graph_def()
    graph_def.ParseFromString(gfile.FastGFile(pb, 'rb').read())
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
    summaryWriter = tf.summary.FileWriter(log_path,graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb', '-p')
    parser.add_argument('--logs', '-l')
    parser.add_argument('--createlogs', '-c', default = True)
    args = parser.parse_args()

    if not args.pb or not args.logs:
        print("Please input as: python tensorboardOfPb.py -p 'model.pb', -l 'logs'")
        print("-p: the path of pb file")
        print("-l: the output path of logs to save")
        exit(1)

    print("\nTips: you can use -c False to not create the logs, the value is default as True\n")

    if args.createlogs == True:
        createLogs(args.pb, args.logs)

    username = os.getcwd().split('/')[2].split('/')[0]
    tensorboard_path = "/home/" + username + "/anaconda3/bin"
    os.chdir(tensorboard_path)
    os.system("./tensorboard --logdir args.logs")
