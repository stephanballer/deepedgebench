from time import time
import numpy as np
from serial import Serial
from time import sleep
from utils.img_utils import load_preproc_images

BUF_SIZE = 256
PAGE_SIZE = int((128 * 128 * 3) / BUF_SIZE)

def inference_arduino(input_dims, dataset, serial_device):
    results, timestamps = list(), list()
    total_time, img_cnt = 0.0, 0

    ser = Serial(serial_device, 9600)

    for img in dataset:
        img_dims, img = load_preproc_images([img], 'int8', (input_dims[0], input_dims[1]))
        img = img[0].flatten().reshape((PAGE_SIZE, BUF_SIZE)).tolist()
        img_cnt += 1
        
        for cnt, arr in enumerate(img):
            print('\x1b[1K\rLoading image: %d/%d' % (cnt+1, PAGE_SIZE), end='')
            ser.write(bytes(arr))
            while(ser.inWaiting() <= 0):
                sleep(0.01)
            ser.read()
        
        inf_start_time = time()
        print()
        
        out = bytes()
        while(len(out) < 1005):
            out += ser.read(ser.inWaiting())
            sleep(0.01)

        duration = int.from_bytes(out[-4:], 'little', signed=False)/1000

        timestamps.append(("inf_start_batch_%d" % (img_cnt), inf_start_time))
        timestamps.append(("inf_end_batch_%d" % (img_cnt), inf_start_time + duration))
    
        # Log
        total_time += duration
        print('Inference took %f seconds' % (duration))
    
        out = np.frombuffer(out[:1001], dtype=np.uint8)
        print(list(filter(lambda x:x < 0, out)))
        results.append(([[out]], img_dims))
    
    print('Inferenced %d images in %f seconds' % (img_cnt, total_time))

        
    ser.close()
        
    return results, timestamps
