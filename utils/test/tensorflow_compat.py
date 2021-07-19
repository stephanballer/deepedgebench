import tensorflow as tf
import tensorflow_hub as hub
from time import time
import numpy as np
from utils.img_utils import load_preproc_images

def inference_tf1(model_path, datatype, input_dims, dataset, batch_size, repeat, tags, input_signature):
    timestamps, results = list(), list()

    # Set up model
    model = hub.load(model_path, tags=set(tags.split(',')))
    model = model.signatures[input_signature]
    
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
    
        # Inference for tensorflow 1 and 2 and append outputs to result
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
