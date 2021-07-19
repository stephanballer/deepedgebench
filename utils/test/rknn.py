from rknn.api import RKNN
from time import time
from utils.img_utils import load_preproc_images
from time import time

def inference_rknn(model_path, datatype, input_dims, dataset, batch_size, repeat):
    timestamps, results = list(), list()

    init_time_start = time()
    rknn = RKNN()
    rknn.load_rknn(path=model_path)
    rknn.init_runtime()
    init_time_end = time()
    timestamps.append(("init_rknn_start", init_time_start))
    timestamps.append(("init_rknn_end", init_time_end))
    print('Initialization took %f seconds' % (init_time_end - init_time_start))
    
    # Inference loop
    total_time, img_cnt = 0.0, 0
    while img_cnt + batch_size <= len(dataset):
        img_paths = dataset[img_cnt:img_cnt + batch_size]
        img_cnt += batch_size
    
        img_dims, input_tensor = load_preproc_images(img_paths, datatype, input_dims)
        input_tensor = [x for x in input_tensor]
    
        # Inference and append output to results
        inf_start_time = time()
        for i in range(repeat):
            out = rknn.inference(inputs=input_tensor)
        inf_end_time = time()
        timestamps.append(("inf_start_batch_%d" % (img_cnt), inf_start_time))
        timestamps.append(("inf_end_batch_%d" % (img_cnt), inf_end_time))
    
        duration = inf_end_time-inf_start_time
        total_time += duration
        print('Inference took %f seconds' % (duration))
    
        #print(list(filter(lambda x: x >= 0.01, out[0][0])), len(out), len(out[0]), len(out[0][0]))
        results.append((out, img_dims))
    
    rknn.release()
    print('Inferenced %d images in %f seconds' % (img_cnt, total_time))

    return results, timestamps
