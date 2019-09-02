import argparse
import tensorflow as tf
import  numpy as np
import time
import cv2
import sys
 
def recognize(w, h, c, inputs, pb, inName, outName):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
 
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            for node in output_graph_def.node :
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

            _ = tf.import_graph_def(output_graph_def, name="")
 
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
 
            input_x = sess.graph.get_tensor_by_name(inName)
            out_softmax = sess.graph.get_tensor_by_name(outName)
 
            img = cv2.imread(inputs)
            img_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            test_img = cv2.resize(img, (w, h))
            test_img = np.asarray(test_img, np.float32)
            test_img = np.reshape(test_img, (1,w,h,c))
            test_img = test_img[np.newaxis, :] / 255.
            
            np.set_printoptions(threshold=sys.maxsize)
            time_start = time.time()
            img_out_softmax = sess.run(out_softmax, feed_dict={input_x:test_img})
            #print(img_out_softmax.shape)

            time_end = time.time()
            print('run time: ', time_end - time_start, 's')
            """
            f = open("weight.txt", "w")
            print(img_out_softmax, file=f)
            f.close
            """
            prediction_labels = np.argmax(img_out_softmax)
            print("label:",prediction_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape_w', '-sw', type=int)
    parser.add_argument('--shape_h', '-sh', type=int)
    parser.add_argument('--shape_c', '-sc', type=int)
    parser.add_argument("--input", '-i')
    parser.add_argument("--pb", '-p')
    parser.add_argument("--inName", '-n')
    parser.add_argument("--outputName", '-o')
    args = parser.parse_args()

    if not args.shape_w or not args.shape_h or not args.shape_c or not args.input or not args.pb or not args.inName or not args.outName:
        print("Please input as: python testOfpb.py -sw 16 -sh 16 -sc 3 -i '1.jpg' -p 'model.pb' -n 'input_x:0' -o 'ouput/fc_1/Matmul:0' ")
        print("-sw/-sh/-sc: shape of input node")
        print("-i: input of the model")
        print("-p: the path of the pb file")
        print("-n: the input node name, note the name should be xxx:0")
        print("-o: the output node name, note the name should be xxx:0")
        exit(1)

    print("\n Warning: you'd better check the detail of this test since the input are preprocess for image like image/255.0\n")

    recognize(args.shape_w, args.shape_h, args.shape_c, args.input, args.pb, args.inName, args.outName)