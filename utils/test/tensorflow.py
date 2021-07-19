import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from time import time
import numpy as np
from utils.img_utils import load_preproc_images

def inference_tf(model_path, datatype, input_dims, dataset, batch_size, repeat):
#with tf.device('/GPU:0'):
    timestamps, results = list(), list()

    # Set up model
    model = tf.saved_model.load(model_path)
    #model = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    
    # Tensorflow ist loading the model on first inference, to test loading time we set some random input
    img_paths = dataset[:batch_size if batch_size <= len(dataset) else -1]
    _, input_tensor = load_preproc_images(img_paths, datatype, input_dims)
    input_tensor = tf.convert_to_tensor(input_tensor)
    init_time_start = time()
    model(input_tensor)
    init_time_end = time()
    timestamps.append(("init_inference_start", init_time_start))
    timestamps.append(("init_inference_end", init_time_end))

    print('First inference took %f seconds' % (init_time_end - init_time_start))
    
    # Inference loop
    total_time, img_cnt = 0.0, 0
    while img_cnt + batch_size <= len(dataset):
        img_paths = dataset[img_cnt:img_cnt + batch_size]
        img_cnt += batch_size
    
        img_dims, input_tensor = load_preproc_images(img_paths, datatype, input_dims)
        input_tensor = tf.convert_to_tensor(input_tensor)
    
        # Inference for tensorflow 2 and append outputs to result
        inf_start_time = time()
        for i in range(repeat):
            out = model(input_tensor)
        inf_end_time = time()
        timestamps.append(("inf_start_batch_%d" % (img_cnt), inf_start_time))
        timestamps.append(("inf_end_batch_%d" % (img_cnt), inf_end_time))
    
        # Log
        duration = inf_end_time-inf_start_time
        total_time += duration
        print('Inference took %f seconds' % (duration))
    
        out = {k:v.numpy() for k, v in out.items()}
        results.append((out, img_dims))
    
    print('Inferenced %d images in %f seconds' % (img_cnt, total_time))

    return results, timestamps


def inference_tf_frozen(model_path, datatype, input_dims, dataset, batch_size, repeat, input_name, output_names):
#with tf.device('/GPU:0'):
    timestamps, results = list(), list()

    # Set up model
    #model = tf.saved_model.load(model_path)
    graph = tf.compat.v1.Graph()
    sess = tf.compat.v1.InteractiveSession(graph=graph)

    with tf.compat.v1.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    datatype = datatype.lower()
    if datatype == 'int8':
        np_type = np.uint8
    elif datatype == 'fp16':
        np_type = np.float16
    elif datatype == 'fp32':
        np_type = np.float32
    else:
        print('Datatype not supported')
        exit(1)

    if input_dims is None:
        graph_input = tf.compat.v1.placeholder(np_type, shape=[batch_size, None, None, 3], name=input_name)
    else:
        graph_input = tf.compat.v1.placeholder(np_type, shape=[batch_size, input_dims[0], input_dims[1], 3], name=input_name)
    tf.compat.v1.import_graph_def(graph_def, {input_name:graph_input})
    output_tensors = list(map(lambda x: graph.get_tensor_by_name('import/'+x), output_names))

    # Tensorflow ist loading the model on first inference, to test loading time we set some random input
    img_paths = dataset[:batch_size]
    _, input_tensor = load_preproc_images(img_paths, datatype, input_dims)
    init_time_start = time()
    sess.run(output_tensors, feed_dict={graph_input:input_tensor})
    init_time_end = time()
    timestamps.append(("init_inference_start", init_time_start))
    timestamps.append(("init_inference_end", init_time_end))

    print('First inference took %f seconds' % (init_time_end - init_time_start))

    #flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    #print('Model needs %s FLOPS after freezing' % (flops.total_float_ops))
    #ops = tf.compat.v1.profiler.profile(graph)
    #print('------')
    #print(ops)
    #print('------')

    # Inference loop
    total_time, img_cnt = 0.0, 0
    while img_cnt + batch_size <= len(dataset):
        img_paths = dataset[img_cnt:img_cnt + batch_size]
        img_cnt += batch_size

        img_dims, input_tensor = load_preproc_images(img_paths, datatype, input_dims)

        # Inference for tensorflow 1 and 2 and append outputs to result
        inf_start_time = time()
        for i in range(repeat):
            out = sess.run(output_tensors, feed_dict={graph_input:input_tensor})
        inf_end_time = time()
        timestamps.append(("inf_start_batch_%d" % (img_cnt), inf_start_time))
        timestamps.append(("inf_end_batch_%d" % (img_cnt), inf_end_time))

        # Log
        duration = inf_end_time-inf_start_time
        total_time += duration
        print('Inference took %f seconds' % (duration))

        results.append((out, img_dims))

    sess.close()
    print('Inferenced %d images in %f seconds' % (img_cnt, total_time))

    return results, timestamps
