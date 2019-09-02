import tensorflow as tf
import numpy as np

"""
(1280 * 640)
input = (640 * 320)
640 * 320
320 * 160
160 * 80
80 * 40
40 * 20
20 * 10
10 * 5
"""
leaky_alpha = 0.1

xavier_initializer = tf.initializers.glorot_uniform()


def conv_block(x, filters, stride, out_channel, net_type, is_training, name='', relu=True):
    """
    :param x: input :nhwc
    :param filters: list [f_w, f_h]
    :param stride: list int
    :param out_channel: int, out_channel
    :param net_type: cnn mobilenet
    :param is_training: used in BN
    :param name: str
    :param relu: boolean
    :return: depwise and pointwise out
    """
    with tf.name_scope('' + name):
        in_channel = x.shape[3].value
        if net_type == 'cnn':
            with tf.name_scope('cnn'):
                # weight = tf.Variable(tf.truncated_normal([filters[0], filters[1], in_channel, out_channel], 0, 0.01))
                weight = tf.Variable(xavier_initializer([filters[0], filters[1], in_channel, out_channel]))
                if stride[0] == 2:  # refer to "https://github.com/qqwweee/keras-yolo3/issues/8"
                    x = tf.pad(x, tf.constant([[0, 0], [1, 0, ], [1, 0], [0, 0]]))
                    x = tf.nn.conv2d(x, weight, [1, stride[0], stride[1], 1], 'VALID')
                else:
                    x = tf.nn.conv2d(x, weight, [1, stride[0], stride[1], 1], 'SAME')
                if relu:
                    x = tf.layers.batch_normalization(x, training=is_training)
                    x = tf.nn.leaky_relu(x, leaky_alpha)
                else:
                    bias = tf.Variable(tf.zeros(shape=out_channel))
                    x += bias
        elif net_type == 'mobilenetv1':
            with tf.name_scope('depthwise'):
                # depthwise_weight = tf.Variable(tf.truncated_normal([filters[0], filters[1], in_channel, 1], 0, 0.01))
                depthwise_weight = tf.Variable(xavier_initializer([filters[0], filters[1], in_channel, 1]))
                x = tf.nn.depthwise_conv2d(x, depthwise_weight, [1, stride[0], stride[1], 1], 'SAME')
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.relu6(x)

            with tf.name_scope('pointwise'):
                # pointwise_weight = tf.Variable(tf.truncated_normal([1, 1, in_channel, out_channel], 0, 0.01))
                pointwise_weight = tf.Variable(xavier_initializer([1, 1, in_channel, out_channel]))
                x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
                if relu:
                    x = tf.layers.batch_normalization(x, training=is_training)
                    x = tf.nn.relu6(x)
                else:
                    bias = tf.Variable(tf.zeros(shape=out_channel))
                    x += bias

        elif net_type == 'mobilenetv2':
            tmp_channel = out_channel * 3
            with tf.name_scope('expand_pointwise'):
                pointwise_weight = tf.Variable(xavier_initializer([1, 1, in_channel, tmp_channel]))
                x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
                x = tf.layers.batch_normalization(x, training=is_training)
                x = tf.nn.relu6(x)
            with tf.name_scope('depthwise'):
                depthwise_weight = tf.Variable(xavier_initializer([filters[0], filters[1], tmp_channel, 1]))
                x = tf.nn.depthwise_conv2d(x, depthwise_weight, [1, stride[0], stride[1], 1], 'SAME')
            with tf.name_scope('project_pointwise'):
                pointwise_weight = tf.Variable(xavier_initializer([1, 1, tmp_channel, out_channel]))
                x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
                if relu:
                    x = tf.layers.batch_normalization(x, training=is_training)
                else:
                    bias = tf.Variable(tf.zeros(shape=out_channel))
                    x += bias
        else:
            raise Exception('net type is error, please check')
    return x


def residual(x, net_type, is_training, out_channel=1, expand_time=1, stride=1):
    if net_type in ['cnn', 'mobilenetv1']:
        out_channel = x.shape[3].value
        shortcut = x
        x = conv_block(x, [1, 1], [1, 1], out_channel // 2, net_type='cnn', is_training=is_training)
        x = conv_block(x, [3, 3], [1, 1], out_channel, net_type='cnn', is_training=is_training)
        x += shortcut

    elif net_type == 'mobilenetv2':
        shortcut = x
        in_channel = x.shape[3].value
        tmp_channel = in_channel * expand_time
        with tf.name_scope('expand_pointwise'):
            pointwise_weight = tf.Variable(xavier_initializer([1, 1, in_channel, tmp_channel]))
            x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.relu6(x)
        with tf.name_scope('depthwise'):
            depthwise_weight = tf.Variable(xavier_initializer([3, 3, tmp_channel, 1]))
            x = tf.nn.depthwise_conv2d(x, depthwise_weight, [1, stride, stride, 1], 'SAME')
        with tf.name_scope('project_pointwise'):
            pointwise_weight = tf.Variable(xavier_initializer([1, 1, tmp_channel, out_channel]))
            x = tf.nn.conv2d(x, pointwise_weight, [1, 1, 1, 1], 'SAME')
            x = tf.layers.batch_normalization(x, training=is_training)
        x += shortcut

    return x


def upsample(x, scale):
    new_height = x.shape[1] * scale
    new_width = x.shape[2] * scale
    resized = tf.image.resize_images(x, [new_height, new_width])
    return resized


def full_yolo_body(x, out_channel, net_type, is_training):
    channel = out_channel
    if net_type in ['mobilenetv2']:
        net_type = 'mobilenetv1'
    x = conv_block(x, [1, 1], [1, 1], channel // 2, net_type, is_training=is_training)
    x = conv_block(x, [3, 3], [1, 1], channel, net_type, is_training=is_training)
    x = conv_block(x, [1, 1], [1, 1], channel // 2, net_type, is_training=is_training)
    x = conv_block(x, [3, 3], [1, 1], channel, net_type, is_training=is_training)
    x = conv_block(x, [1, 1], [1, 1], channel // 2, net_type, is_training=is_training)
    x_route = x
    x = conv_block(x, [3, 3], [1, 1], channel, net_type, is_training=is_training)
    return x_route, x


def full_darknet_body(x, net_type, is_training):
    """
    yolo3_tiny build by net_type
    :param x:
    :param is_training:
    :param net_type: cnn mobilenet
    :return:
    """
    if net_type in ['cnn', 'mobilenetv1']:
        x = conv_block(x, [3, 3], [1, 1], 32, 'cnn', is_training=is_training)

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 64, 'cnn', is_training=is_training)
        for i in range(1):
            x = residual(x, net_type, is_training)

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 128, 'cnn', is_training=is_training)
        for i in range(2):
            x = residual(x, net_type, is_training)

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 256, 'cnn', is_training=is_training)
        for i in range(8):
            x = residual(x, net_type, is_training)
        route2 = x

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 512, 'cnn', is_training=is_training)
        for i in range(8):
            x = residual(x, net_type, is_training)
        route1 = x

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 1024, 'cnn', is_training=is_training)
        for i in range(4):
            x = residual(x, net_type, is_training)

    elif net_type == 'mobilenetv2':
        # down sample
        x = conv_block(x, [3, 3], [2, 2], 32, 'cnn', is_training=is_training)

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 64, net_type, is_training=is_training)
        for i in range(2):
            x = residual(x, net_type, is_training, 64, 1)

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 96, net_type, is_training=is_training)
        for i in range(4):
            x = residual(x, net_type,is_training, 96, 6)
        route2 = x

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 160, net_type, is_training=is_training)
        for i in range(4):
            x = residual(x, net_type, is_training, 160, 6)
        route1 = x

        # down sample
        x = conv_block(x, [3, 3], [2, 2], 320, net_type, is_training=is_training)
        for i in range(3):
            x = residual(x, net_type, is_training, 320, 1)

    else:
        route1, route2 = [], []
    return x, route1, route2


def full_yolo_head(x, route1, route2, num_class, anchors, net_type, is_training):
    with tf.name_scope('body_layer1'):
        x_route, x = full_yolo_body(x, 1024, net_type, is_training)
    x = conv_block(x, [1, 1], [1, 1], 3 * (5 + num_class), 'cnn', is_training,  "yolo_head1", False)
    output1 = x
    #box1 = yolo(x, anchors[[6, 7, 8]])

    with tf.name_scope('head_layer2'):
        x = conv_block(x_route, [1, 1], [1, 1], x_route.shape[-1].value // 2, net_type, is_training)
        x = upsample(x, 2)
        x = tf.concat([x, route1], 3)
        x_route, x = full_yolo_body(x, 512, net_type, is_training)
    x = conv_block(x, [1, 1], [1, 1], 3 * (5 + num_class), 'cnn', is_training, "yolo_head2", False)
    output2 = x
    #box2= yolo(x, anchors[[3, 4, 5]])

    with tf.name_scope('head_layer3'):
        x = conv_block(x_route, [1, 1], [1, 1], x_route.shape[-1].value // 2, net_type, is_training)
        x = upsample(x, 2)
        x = tf.concat([x, route2], 3)
        x_route, x = full_yolo_body(x, 256, net_type, is_training)
    x = conv_block(x, [1, 1], [1, 1], 3 * (5 + num_class), 'cnn', is_training, "yolo_head3", False)
    output3 = x
    #box3 = yolo(x, anchors[[0, 1, 2]])

    #boxes = tf.concat([box1, box2, box3], 1)
    return output1, output2, output3


def tiny_darknet_body(x, net_type, is_training):
    """
    yolo3_tiny build by net_type
    :param x:
    :param is_training: used in bn
    :param net_type: cnn or mobile-net
    :return:
    """
    if net_type in ['mobilenetv1', 'mobilenetv2']:
        net_type = 'mobilenetv1'
    x = conv_block(x, [3, 3], [1, 1], 16, net_type, is_training)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    x = conv_block(x, [3, 3], [1, 1], 32, net_type, is_training)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    x = conv_block(x, [3, 3], [1, 1], 64, net_type, is_training)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    x = conv_block(x, [3, 3], [1, 1], 128, net_type, is_training)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    x = conv_block(x, [3, 3], [1, 1], 256, net_type, is_training)
    x_route = x
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    x = conv_block(x, [3, 3], [1, 1], 512, net_type, is_training)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')

    x = conv_block(x, [3, 3], [1, 1], 1024, net_type, is_training)

    return x, x_route


def tiny_yolo_head(x, x_route1, num_class, anchors, net_type, is_training):
    with tf.name_scope('head_layer1'):
        x = conv_block(x, [1, 1], [1, 1], 256, net_type, is_training)
        x_route2 = x
        x = conv_block(x, [3, 3], [1, 1], 512, net_type, is_training)
        x = conv_block(x, [1, 1], [1, 1], 3 * (5 + num_class), 'cnn', is_training, "yolo_head1", False)
        fe1 = x
        fe1, box1, grid1 = yolo(fe1, anchors[[3, 4, 5]])

    with tf.name_scope('head_layer2'):
        x = conv_block(x_route2, [1, 1], [1, 1], 128, net_type, is_training)
        x = upsample(x, 2)
        x = tf.concat([x, x_route1], 3)
        x = conv_block(x, [3, 3], [1, 1], 256, net_type, is_training)
        x = conv_block(x, [1, 1], [1, 1], 3 * (5 + num_class), 'cnn', is_training, "yolo_head2", False)
        fe2 = x
        fe2, box2, grid2 = yolo(fe2, anchors[[0, 1, 2]])

    fe = tf.concat([fe1, fe2], 1)
    box = tf.concat([box1, box2], 1)
    return fe, box, grid1, grid2


def yolo(f, anchors):
    """
    convert feature to box and scores
    :param f:
    :param anchors:
    :return:
    """

    anchor_tensor = tf.constant(anchors, tf.float32)
    f = tf.reshape(f, [1, f.shape[1], f.shape[2], 3, 85])
    grid_y = tf.tile(tf.reshape(tf.range(f.shape[1]), [1, -1, 1, 1]), [1, 1, f.shape[2], 1])
    grid_x = tf.tile(tf.reshape(tf.range(f.shape[2]), [1, 1, -1, 1]), [1, f.shape[1], 1, 1])
    grid = tf.tile(tf.cast(tf.concat([grid_x, grid_y], -1), tf.float32)[:, :, :, tf.newaxis, :], (1, 1, 1, 3, 1))

    box_xy = (tf.nn.sigmoid(f[..., :2]) + grid) / tf.cast(grid.shape[::-1][2:4], tf.float32, )
    box_wh = tf.math.exp(f[..., 2:4]) * anchor_tensor
    box_confidence = tf.nn.sigmoid(f[..., 4:5])
    classes_score = tf.nn.sigmoid(f[..., 5:])
    boxes = tf.reshape(tf.concat([box_xy, box_wh, box_confidence, classes_score], -1), [1, -1, 3, f.shape[4]])

    return boxes


def model(x, num_classes, net_type, is_training):
    path = "./yolo_anchors.txt"
    with open(path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)

    #batchsize, height, width, _ = x.get_shape().as_list()
    """
    if len(anchors) == 6:
        x, x_route = tiny_darknet_body(x, net_type, is_training)
        raw_pred, y, *grid = tiny_yolo_head(x, x_route, num_classes, anchors, net_type, is_training)
    else:
        x, route1, route2 = full_darknet_body(x, net_type, is_training)
        raw_pred, y, *grid = full_yolo_head(x, route1, route2, num_classes, anchors, net_type, is_training)
    """
    x, route1, route2 = full_darknet_body(x, net_type, is_training=False)
    output1, output2, output3 = full_yolo_head(x, route1, route2, num_classes, anchors, net_type, is_training=False)
    
    #box_xy, box_wh, box_confidence, classes_score = y[..., :2], y[..., 2:4], y[..., 4:5], y[..., 5:]
    #box_xy *= tf.constant([416, 416], tf.float32)
    # box_wh *= tf.constant([width, height], tf.float32)
    #boxe = tf.concat([box_xy, box_wh, box_confidence, classes_score], -1, name='debug_pred')

    return output1, output2, output3


if __name__ == '__main__':

    restore_path = "./models/cnn_full_model_epoch_16"
    class_num = 80

    with tf.Session() as sess:
        input_x = tf.placeholder(tf.float32, shape=[None, 416, 416, 3], name='input_x')
        output1, output2, output3= model(input_x, class_num, 'cnn', False)

        saver = tf.train.Saver()
        saver.restore(sess, restore_path)

        # generate graph

        tf.train.write_graph(sess.graph.as_graph_def(), '.', './models/yolov3.pbtxt', as_text=True)
