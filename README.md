######################################
tf2ncnn by ziyang
######################################

Warning: All below related config like file path in every file should be changed first.

First: Turn the ckpt files of the saved model to pbtxt

Before training the model you defined, you'd better make sure the type and shape of the input, also the output of the model, 
as in ckpt2pbtxt.py, line 12, 13

	input_x = tf.placeholder(tf.float32, shape=[None, 56,56, 3], name='input_x')
	logits = classify_model(input_x, class_num)

  the line above requests to be changed according to your model, for more information please check in ckpt2pbtxt.py. Make sure
the related ckpt files for the trained model are saved correctly and the related changes are made in ckpt2pbtxt.py.  Then , run the 
follow cmd:
	
	python ckpt2pbtxt.py

Second: Turn pbtxt to pb

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

Third: Turn pb to the param and bin files for ncnn

1. Run the follow cmd:

	./tensorflow2ncnn xxx.pb ncnn.param ncnn.bin

2. Check the model defined in ncnn.param, the model name and other information can be read in ncnn.param,
the input and output name will be used for build the model in ncnn.


For more information of covert2pb.py, please check:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py

def freeze_graph(input_graph,
                 input_saver,
                 input_binary,
                 input_checkpoint,
                 output_node_names,
                 restore_op_name,
                 filename_tensor_name,
                 output_graph,
                 clear_devices,
                 initializer_nodes,
                 variable_names_whitelist="",
                 variable_names_blacklist="",
                 input_meta_graph=None,
                 input_saved_model_dir=None,
                 saved_model_tags=tag_constants.SERVING,
                 checkpoint_version=saver_pb2.SaverDef.V2):
  """Converts all variables in a graph and checkpoint into constants.
  Args:
    input_graph: A `GraphDef` file to load.
    input_saver: A TensorFlow Saver file.
    input_binary: A Bool. True means input_graph is .pb, False indicates .pbtxt.
    input_checkpoint: The prefix of a V1 or V2 checkpoint, with V2 taking
      priority.  Typically the result of `Saver.save()` or that of
      `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or
      V1/V2.
    output_node_names: The name(s) of the output nodes, comma separated.
    restore_op_name: Unused.
    filename_tensor_name: Unused.
    output_graph: String where to write the frozen `GraphDef`.
    clear_devices: A Bool whether to remove device specifications.
    initializer_nodes: Comma separated list of initializer nodes to run before
                       freezing.
    variable_names_whitelist: The set of variable names to convert (optional, by
                              default, all variables are converted),
    variable_names_blacklist: The set of variable names to omit converting
                              to constants (optional).
    input_meta_graph: A `MetaGraphDef` file to load (optional).
    input_saved_model_dir: Path to the dir with TensorFlow 'SavedModel' file and
                           variables (optional).
    saved_model_tags: Group of comma separated tag(s) of the MetaGraphDef to
                      load, in string format.
    checkpoint_version: Tensorflow variable file format (saver_pb2.SaverDef.V1
                        or saver_pb2.SaverDef.V2).
  Returns:
    String that is the location of frozen GraphDef.
