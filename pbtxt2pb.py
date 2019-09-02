import argparse
from tensorflow.python.tools import freeze_graph

def turnPb(pbtxt, ckpt, output_node, output_path):
    freeze_graph.freeze_graph(pbtxt, "", False, ckpt, output_node,
                            "save/restore_all", "save/Const:0", output_path, True, "")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pbtxt', '-p')
    parser.add_argument('--ckpt', '-c')
    parser.add_argument('--output_node', '-n')
    parser.add_argument('--output_path', '-o')
    args = parser.parse_args()

    if not args.pbtxt or not args.ckpt or not args.output_node or not args.output_path:
        print("Please input as: python pbtxt2pb.py -p 'models/model.pbtxt' -c 'model_epoch_100' -n 'output/cnn/add' -o 'output.pb'")
        print("-p: the path of the pbtxt")
        print("-c: the path of all related ckpt files(end without .ckpt/meta...")
        print("-n: the node of the desired output node")
        print("-o: the path of the converted pb file to saved")
        exit(1)

    turnPb(args.pbtxt, args.ckpt, args.output_node , args.output_path)
    print("Finished")

    #python pbtxt2pb.py -p 'models/yolov3.pbtxt' -c 'cnn_full_model_epoch_42' -n 'yolo_head3/cnn/add' -o 'yolov3.pb'