import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
from tensorflow.python.saved_model import tag_constants
from time import time
from threading import Thread, Event
from queue import Queue
from PIL import Image, ImageDraw, ImageFont

# Draws a list of bounding boxes and labels to image and cycle through
# given colors
def draw_bounding_boxes(img, bboxs, labels, colors):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for i, (bbox, label) in enumerate(zip(bboxs, labels)):
        draw.rectangle(bbox, outline=colors[i % len(colors)])
        text_width, text_height = font.getsize(label)
        draw.rectangle((bbox[0], bbox[3]-text_height, bbox[0]+text_width, bbox[3]), fill=colors[i % len(colors)],
                outline=colors[i % len(colors)])
        draw.text((bbox[0], bbox[3] - text_height), label, fill=(0,0,0), font=font)

    return img



model_path, input_dims = \
    '../models/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model', (300, 300)

with tf.device('/GPU:0'):
    # Set up model
    #model = tf.saved_model.load(model_path)
    model = hub.load(model_path, tags='serve')
    model = model.signatures['serving_default']
    
    img_cnt = 0
    def inference(img):
        global img_cnt
        img_cnt += 1
    
        img_dims, input_tensor = img.shape, np.expand_dims(cv2.resize(img, input_dims, cv2.INTER_AREA).astype(np.uint8), 0)
        input_tensor = tf.convert_to_tensor(input_tensor)
    
        out = model(input_tensor)
    
        return img_dims, out    
    
    #def gstreamer_pipeline(
    #    capture_width=1280,
    #    capture_height=720,
    #    display_width=1280,
    #    display_height=720,
    #    framerate=60,
    #    flip_method=0,
    #):
    #    return (
    #        "nvarguscamerasrc ! "
    #        "video/x-raw(memory:NVMM), "
    #        "width=(int)%d, height=(int)%d, "
    #        "format=(string)NV12, framerate=(fraction)%d/1 ! "
    #        "nvvidconv flip-method=%d ! "
    #        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
    #        "videoconvert ! "
    #        "video/x-raw, format=(string)BGR ! appsink"
    #        % (
    #            capture_width,
    #            capture_height,
    #            framerate,
    #            flip_method,
    #            display_width,
    #            display_height,
    #        )
    #    )
    
    #cam = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    cam = cv2.VideoCapture(2)
    #fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
    stream_fps = 1
    #stream = cv2.VideoWriter('stream.avi', fourcc, stream_fps, (1280,720))
    def quit(event):
        input('Press any key to exit')
        event.set()

    def inference_worker(event, queue):
        while not event.is_set():
            t_cam = time()
            ret, frame = cam.read()
            t_cam = time() - t_cam
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_inf = time()
            img_dims, result = inference(frame)
            t_inf = time() - t_inf
            queue.put((frame, t_cam, t_inf, img_dims, result))

    cevent, wevent = Event(), Event()
    queue = Queue(1)
    worker = Thread(target=inference_worker, args=[wevent, queue])
    worker.start()
    Thread(target=quit, args=[cevent]).start()
    
    # Load coco result dataset file and print summary
    with open('../datasets/coco_labels.txt', 'r') as f:
        labels = f.read().split('\n')
    
    print()
    timestamp = time()
    offset = 0.0
    while not cevent.is_set():    
        frame, t_cam, t_inf, img_dims, result = queue.get()

        t_eval = time()
        scores = result['detection_scores'][0]
        bboxs = [[e[1] * img_dims[1], e[0] * img_dims[0], e[3] * img_dims[1], e[2] * img_dims[0]] for e in result['detection_boxes'][0]]
        cats = [labels[int(i)] if int(i) < len(labels) else labels[0] for i in result['detection_classes'][0]]
        filter_list = list(zip(*filter(lambda x: x[0] > 0.5, zip(scores, bboxs, cats))))
        if len(filter_list) > 0:
            bboxs, cats = filter_list[1:]
            img = draw_bounding_boxes(Image.fromarray(frame), bboxs, cats, [(0, 191, 255)])
            frame = np.array(img)
        cur_time = time()
        duration = cur_time - timestamp
        fps = 1.0/duration
        cv2.imwrite('img.jpg', frame)
        t_eval = time() - t_eval
    #    for i in range(int(duration * stream_fps)):
    #        stream.write(frame)
        cv2.imshow('cam', frame)
        cv2.waitKey(1)
    
        print('\x1b[1K\r%.2fFPS/%.3fs cam: %.3fs inf: %.3fs eval: %.3fs' % (fps, duration, t_cam, t_inf, t_eval), end='')
        duration += offset
        offset = duration - float(int(duration))
        timestamp = cur_time

    wevent.set()
    if queue.full():
        queue.get(block=False)
    worker.join()
    cam.release()
