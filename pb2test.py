import tensorflow as tf
import  numpy as np
import time
import cv2
 
def recognize(jpg_path, pb_file_path):
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
 
            input_x = sess.graph.get_tensor_by_name("input_x:0")
            out_softmax = sess.graph.get_tensor_by_name("comExp/ouput/Conv2D:0")
 
            img = cv2.imread(jpg_path)
            img_ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            test_img = cv2.resize(img_ori, (56, 56))
            test_img = np.asarray(test_img, np.float32)
            test_img = np.reshape(test_img, (1,56,56,3))
            #test_img = test_img[np.newaxis, :] / 255.
 
            time_start = time.time()
            img_out_softmax = sess.run(out_softmax, feed_dict={input_x:test_img})
            img_out_softmax = np.reshape(img_out_softmax, [-1, 7])
            time_end = time.time()
            print('run time: ', time_end - time_start, 's')
 
            print("img_out_softmax:",img_out_softmax)
            prediction_labels = np.argmax(img_out_softmax)
            print("label:",prediction_labels)
 
recognize('1.jpg', "./models/exp.pb")