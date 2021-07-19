import cv2
import numpy as np
import tensorflow as tf
from utils.imagenet_utils import *
from time import time

if __name__ == '__main__':
    model_path, input_dims, input_name, output_names = \
        'models/mobilenet_v2/model.pb', (224, 224), 'input', ['MobilenetV2/Predictions/Reshape_1:0']
    #with tf.device('/GPU:0'):
    timestamps = list()

    # Set up model
    #model = tf.saved_model.load(model_path)
    graph = tf.compat.v1.Graph()
    sess = tf.compat.v1.InteractiveSession(graph=graph)

    with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    if input_dims is None:
        graph_input = tf.compat.v1.placeholder(np.float32, shape=[1, None, None, 3], name=input_name)
    else:
        graph_input = tf.compat.v1.placeholder(np.float32, shape=[1, input_dims[0], input_dims[1], 3], name=input_name)
    tf.compat.v1.import_graph_def(graph_def, {'input':graph_input})
    output_tensors = list(map(lambda x: graph.get_tensor_by_name('import/'+x), output_names))

    #flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    #print('Model needs %s FLOPS after freezing' % (flops.total_float_ops))
    #ops = tf.compat.v1.profiler.profile(graph)
    #print('------')
    #print(ops)
    #print('------')

    img_cnt = 0
    def inference(img):
        global img_cnt
        img_cnt += 1

        # Inference loop
        img_dims, input_tensor = img.shape, np.expand_dims((cv2.resize(img, input_dims, cv2.INTER_AREA).astype(np.float32)/255.0) - 0.5, 0)

        # Inference for tensorflow 1 and 2 and append outputs to result
        inf_start_time = time()
        out = sess.run(output_tensors, feed_dict={graph_input:input_tensor})
        inf_end_time = time()
        timestamps.append(("inf_start_batch_%d" % (img_cnt), inf_start_time))
        timestamps.append(("inf_end_batch_%d" % (img_cnt), inf_end_time))

        return out, img_dims


    def gstreamer_pipeline(
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=60,
        flip_method=0,
    ):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
        )

    cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    labels = load_labels('datasets/imagenet_2012_labels.txt')
    while True:
        ret, frame = cam.read()
        out = inference(frame)
        results = imagenet_label([out], labels)
        if len(results) > 0:
            print(results)

    sess.close()
    cam.release()
