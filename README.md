# ckpt2pb2ncnn

Warning: All below related config like file path in every file should be changed first.

## First: Turn the ckpt files of the saved model to pbtxt

Before training the model you defined, you'd better make sure the type and shape of the input, also the output of the model, 
as in ckpt2pbtxt.py, line 12, 13

	input_x = tf.placeholder(tf.float32, shape=[None, 56,56, 3], name='input_x')
	logits = classify_model(input_x, class_num)

  the line above requests to be changed according to your model, for more information please check in ckpt2pbtxt.py. Make sure
the related ckpt files for the trained model are saved correctly and the related changes are made in ckpt2pbtxt.py.  Then , run the 
follow cmd:
	
	python ckpt2pbtxt.py

## Second: Turn pbtxt to pb

1. Obtain the input and output node of the model, check in the xxx.pbtxt. Usually, the ouput node is the last node before the node
named startwith save/..., like:

	node{
		name: "comExp/output/Conv2D"  
		#comExp is the defined variables_scope, output is the name defined for output layer, 
		#Conv2d is the corresponding op.
		op: "Conv2D"
		input: "comExp/MaxPool_2"
	... ...


2. Based the name of output node in xxx.pbtxt, run the follow cmd to finish the turn op:
	
	python convert2pb.py

3. To visual or test the .pb files, run the follow cmds:

	python pb2visual.py
	python  pb2test.py

make sure the built .pb files can be tested successfully.

And with the pb2Node.py, you can check  the node in the generated result.txt. The node also can be obtain by 
running the follow cmd based on the ckpt files:

	python getNode.py

## Third: Turn pb to the param and bin files for ncnn

1. Run the follow cmd:

	./tensorflow2ncnn xxx.pb ncnn.param ncnn.bin

2. Check the model defined in ncnn.param, the model name and other information can be read in ncnn.param,
the input and output name will be used for build the model in ncnn.


For more information of covert2pb.py, please check:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
