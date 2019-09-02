import argparse
from tensorflow.python import pywrap_tensorflow

def readNode(checkpoint_path):
	reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
	var_to_shape_map = reader.get_variable_to_shape_map()
	for key in var_to_shape_map:
	    print("tensor_name: ", key)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', '-c')
    args = parser.parse_args()
    if not args.ckpt:
    	print("Please input as: python nodeOfCkpt.py -c 'models/model_epoch_100'")
    	print("-c: the file of ckpt related files")
    	exit(1)

    readNode(args.ckpt)
