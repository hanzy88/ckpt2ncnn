
import tensorflow as tf
from model import *

def classify_model(images, class_num):
    output = comExp(images, class_num)
    return output

restore_path = "./models/expModel.ckpt"
class_num = 7

with tf.Session() as sess:
    input_x = tf.placeholder(tf.float32, shape=[None, 56,56, 3], name='input_x')
    logits = classify_model(input_x, class_num)

    saver = tf.train.Saver()
    saver.restore(sess, restore_path)

    # generate graph

    tf.train.write_graph(sess.graph.as_graph_def(), '.', './models/exp.pbtxt', as_text=True)
