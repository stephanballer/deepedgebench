#!/usr/bin/python3

import argparse
import os
import json
from time import time

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser('Run some saved_model with image as input on a specific platform')
    subparsers = parser.add_subparsers(dest='platform')
    rknnparser = subparsers.add_parser('rknn', help='Run inference on any platform with rknn-toolkit installed')
    tfparser = subparsers.add_parser('tensorflow', help='Run saved model on any platform with tensorflow 2 installed')
    tffrozenparser = subparsers.add_parser('tensorflow_frozen', help='Run frozen graph on any platform with tensorflow 2 installed')
    tf1parser = subparsers.add_parser('tensorflow_1', help='Run saved model on any platform with tensorflow 2 installed in compatibility mode')
    tfliteparser = subparsers.add_parser('tflite', help='Run inference on any platform with tflite_runtime installed')
    onnxparser = subparsers.add_parser('onnx_trt', help='Run inference on any platform with tensorrt installed')
    arduinoparser = subparsers.add_parser('arduino', help='Run inference on arduino')

    parser.add_argument('model_path', help='Path to the model or device if arduino')
    parser.add_argument('dataset_path', help='Path to coco or imagenet dataset')
    parser.add_argument('-t', '--input_type', default='int8', help='Input type (\'FP32\', \'FP16\', \'INT8\'')
    parser.add_argument('-d', '--input_dims', help='Width and height of the input image if not dynamic')
    parser.add_argument('-s', '--score', type=float, default=0.0, help='Minimum score for a detection to be added to output')
    parser.add_argument('-r', '--repeat', type=int, default=1, help='Repeat inference to get more accurate measurements')
    parser.add_argument('-f', '--file', help='Write benchmark data to file with give path')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--num_images', type=int, help='Number of images to inference')
    parser.add_argument('-o', '--save_output', action='store_true', default=False, help='Save images for comparison in folder')
    parser.add_argument('-rd', '--result_dir', default='/tmp/results.json', help='directory to save results to')

    tf1parser.add_argument('-it', '--input_tags', default='serve', help='Input tags, default serve')
    tf1parser.add_argument('-is', '--input_signature', default='serving_default', help='Input signature, default serving_default')

    tffrozenparser.add_argument('-in', '--input_node', required=True, help='Input node of frozen graph')
    tffrozenparser.add_argument('-on', '--output_nodes', required=True, help='Output nodes of frozen graph')

    tfliteparser.add_argument('-u', '--tpu', action='store_true', default=False, help='Use Edge TPU')

    args = parser.parse_args()


    # Load COCO Dataset for object detection (downloads images from web)
    if '.json' in args.dataset_path:
        from utils.coco_utils import *

        coco_gt = COCO(args.dataset_path)
        if args.num_images is not None:
            dataset_coco = coco_gt.loadImgs(coco_gt.getImgIds()[:args.num_images])
        else:
            dataset_coco = coco_gt.loadImgs(coco_gt.getImgIds())
        dataset = [img['coco_url'] for img in dataset_coco]

        # Optional when images are saved locally::
        #dataset_dir = os.path.dirname(os.path.realpath(args.dataset_path))
        #dataset = [os.path.join(dataset_dir, img['file_name']) for img in dataset_coco]

    # Load ImageNet Dataset for image classification (images stored locally)
    elif '.txt' in args.dataset_path:
        dataset_dir = os.path.dirname(os.path.realpath(args.dataset_path))

        with open(args.dataset_path, 'r') as f:
            dataset, ground_truths = list(), list()
            for line in f.readlines():
                filename, gt = line.split()
                dataset.append(os.path.join(dataset_dir, filename))
                ground_truths.append(int(gt))

                if args.num_images is not None and len(dataset) >= args.num_images:
                    break
    else:
        print('Dataset not supported')


    # Set defaults and check arguments
    if args.num_images is None:
        args.num_images = len(dataset)
    if args.input_dims is not None:
        args.input_dims = args.input_dims.split(',')
        if len(args.input_dims) == 1:
            args.input_dims = (int(args.input_dims[0]), int(args.input_dims[0]))
        else:
            args.input_dims = (int(args.input_dims[0]), int(args.input_dims[1]))
    if args.num_images % args.batch_size != 0:
        print("Error: Invalid batch size")
        exit(1)
    if args.num_images == 0:
        print("Error: Invalid number of images")
        exit(1)
        

    # Inference
    # "results" is of type [(output, [(int, int)])] with "output" of shape (output_tensor_number, batch_size)
    test_start = time()
    if args.platform == 'rknn':
        from utils.test.rknn import inference_rknn
        results, timestamps = inference_rknn(args.model_path,
                args.input_type, args.input_dims, dataset, args.batch_size, args.repeat)


    elif args.platform == 'tensorflow':
        from utils.test.tensorflow import inference_tf
        results, timestamps = inference_tf(args.model_path,
                args.input_type, args.input_dims, dataset, args.batch_size, args.repeat)

        # Convert returned dicts to lists in the right order
        if '.json' in args.dataset_path:
            refac = (lambda out: [out['detection_boxes'], out['detection_classes'], out['detection_scores']])
            results = [(refac(out), img_dims) for out, img_dims in results]
        else:
            results = [(list(out.values()), img_dims) for out, img_dims in results]


    elif args.platform == 'tensorflow_1':
        from utils.test.tensorflow_compat import inference_tf1
        results, timestamps = inference_tf1(args.model_path, args.input_type, args.input_dims,
                dataset, args.batch_size, args.repeat, args.input_tags, args.input_signature)

        # Convert returned dicts to lists in the right order
        if '.json' in args.dataset_path:
            refac = (lambda out: [out['detection_boxes'], out['detection_classes'], out['detection_scores']])
            results = [(refac(out), img_dims) for out, img_dims in results]
        else:
            results = [(list(out.values()), img_dims) for out, img_dims in results]


    elif args.platform == 'tensorflow_frozen':
        from utils.test.tensorflow import inference_tf_frozen
        args.output_nodes = args.output_nodes.split(',')
        results, timestamps = inference_tf_frozen(args.model_path, args.input_type, args.input_dims,
                dataset, args.batch_size, args.repeat, args.input_node, args.output_nodes)
        
    elif args.platform == 'tflite':
        from utils.test.tflite import inference_tflite
        results, timestamps = inference_tflite(args.model_path, args.input_type,
                args.input_dims, dataset, args.batch_size, args.repeat, args.tpu)
 
        # Correct classes tensor
        if '.json' in args.dataset_path:
            def refac(out):
                out[1] = out[1] + 1
                return out
            results = [(refac(out), img_dims) for out, img_dims in results]


    elif args.platform == 'onnx_trt':
        from utils.test.onnx_trt import inference_onnx_rt
        results, timestamps = inference_onnx_rt(args.model_path, args.input_type,
                args.input_dims, dataset, args.batch_size, args.repeat)


    elif args.platform == 'arduino':
        from utils.test.arduino import inference_arduino
        results, timestamps = inference_arduino(args.input_dims, dataset, args.model_path)


    else:
        print("Error: mode not supported")
        exit(1)

    test_end = time()
    timestamps.insert(0, ("test_start", test_start))
    timestamps.append(("test_end", test_end))
   
    # Process coco data
    if '.json' in args.dataset_path:
        # Write results in coco format
        res_path = args.result_dir
        write_coco_results(results, res_path, dataset_coco, args.score)

        # Load coco result dataset file and print summary
        coco_det = coco_gt.loadRes(res_path)

        stats = coco_eval(coco_gt, coco_det, dataset_coco)

        # Draw bounding boxes and save them to images folder
        if args.save_output:
            os.makedirs('coco_cmp', exist_ok=True)
            colors_gt = [(255, 255, 255)]
            colors_det = [(0, 191, 255)]
            save_compare_coco_imgs([coco_gt, coco_det], dataset_coco, 'coco_cmp', [colors_gt, colors_det])

        if args.result_dir == '/tmp/results.json':
            os.remove(args.result_dir)

        timestamps.append(('_'.join([str(x) for x in stats]), time()))

    # Process imagenet data
    else:
        from utils.imagenet_utils import *
        # Calculate Top-1 and Top-5 accuracy
        top_1, top_5 = imagenet_eval(results, ground_truths)
        timestamps.append(("top_1_%f__top_5_%f" % (top_1, top_5), time()))
        #print("Top 1: %.3f, Top 5: %.3f" % (top_1, top_5))
        top_1, top_2, eval_labels = imagenet_eval(results, ground_truths, load_labels('datasets/imagenet_2012_labels.txt'))
        for x in eval_labels:
            print(x)
        print("Top 1: %.3f, Top 5: %.3f" % (top_1, top_5))

    # Write timestamps to file. The file can then later be read by serial_reader.py to plot data
    if args.file is not None:
        with open(args.file, 'w') as f:
            f.write('label_data\n')
            for label, time in timestamps:
                f.write('%f %s\n' % (time, label))
