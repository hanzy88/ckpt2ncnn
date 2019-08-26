from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph('./models/exp.pbtxt', "", False,
                        "./models/expModel.ckpt", "comExp/ouput/Conv2D",
                        "save/restore_all", "save/Const:0", "./models/exp.pb", True, "")
