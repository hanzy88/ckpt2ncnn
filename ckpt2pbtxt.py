
import argparse
import tensorflow as tf
from models import *

def modelOutput(images, class_num):
    output = model(images, class_num)
    return output

def turnPbtxt(shapes_w, shapes_h, shapes_c, input_node, class_num, restore_path,  store_path):
	with tf.Session() as sess:
	    input_x = tf.placeholder(tf.float32, shape=[None, shapes_w, shapes_h, shapes_c], name=input_node)
	    logits = modelOutput(input_x, class_num)

	    saver = tf.train.Saver()
	    saver.restore(sess, restore_path)

	    # generate graph

	    tf.train.write_graph(sess.graph.as_graph_def(), '.', store_path, as_text=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--shape_w', '-sw', type=int)
	parser.add_argument('--shape_h', '-sh', type=int)
	parser.add_argument('--shape_c', '-sc', type=int)
	parser.add_argument('--input_node', '-i')
	parser.add_argument('--class_num', '-c')
	parser.add_argument('--restore_path', '-r')
	parser.add_argument('--store_path', '-p')
	args = parser.parse_args()

	if not args.shape_w or not args.shape_h or not args.shape_c or not args.input_node or not args.class_num or not args.restore_path or not args.store_path:
		print("Please input as: python ckpt2pbtxt.py -sw 16 -sh 16 -sc 3 -i 'input_x' -c 7 -r 'models/model_epoch_100' -p 'models/model.pbtxt'")
		print("-sw/-sh/-sc: shape of input node")
		print("-i: name of the input node")
		print("-cls: class number")
		print("-r: place of the file of the stored ckpt")
		print("-p: place of the file of pbtxt to store")
		exit(1)

	turnPbtxt(args.shape_w, args.shape_h, args.shape_c, args.input_node, args.class_num, args.restore_path, args.store_path)
	print("Finished")