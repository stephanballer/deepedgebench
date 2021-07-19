import tflite_runtime.interpreter as tflite
from utils.img_utils import load_preproc_images
from time import time

def inference_tflite(model_path, datatype, input_dims, dataset, batch_size, repeat, tpu=False):
    timestamps, results = list(), list()

    # Set up interpreter
    if tpu:
        interpreter = tflite.Interpreter(model_path=model_path,
                experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    else:
        interpreter = tflite.Interpreter(model_path=model_path)
    
    #print(list(filter(lambda x: 'Conv' not in x, [node['name'] for node in interpreter.get_tensor_details()])))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Tflite ist loading the model on first inference, to test loading time we set some random input
    init_time_start = time()
    interpreter.invoke()
    init_time_end = time()
    timestamps.append(("init_inference_start", init_time_start))
    timestamps.append(("init_inference_end", init_time_end))

    print('First inference took %f seconds' % (init_time_end - init_time_start))
    
    # Inference loop
    total_time, img_cnt = 0.0, 0
    while img_cnt + batch_size <= len(dataset):
        # Load batch
        img_paths = dataset[img_cnt:img_cnt + batch_size]
        img_cnt += batch_size
    
        img_dims, input_tensor = load_preproc_images(img_paths, datatype, input_dims)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
    
        # Inference and append output to results
        inf_start_time = time()
        for i in range(repeat):
            interpreter.invoke()
        inf_end_time = time()
        timestamps.append(("inf_start_batch_%d" % (img_cnt), inf_start_time))
        timestamps.append(("inf_end_batch_%d" % (img_cnt), inf_end_time))
    
        # Log
        duration = inf_end_time-inf_start_time
        total_time += duration
        print('Inference took %f seconds' % (duration))
    
        out = [interpreter.get_tensor(out['index']) for out in output_details]
        results.append((out, img_dims))
    
    print('Inferenced %d images in %f seconds' % (img_cnt, total_time))

    return results, timestamps
