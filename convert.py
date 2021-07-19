#!/usr/bin/python3

import argparse

# Shell command for tf_tftrt:
#  saved_model_cli convert --dir <saved_model_path> --output_dir <output_model_path> --tag_set serve --precision_mode FP16 tensorrt
# Shell command for edge TPU with docker:
# docker build --tag edgetpu_compiler https://github.com/tomassams/docker-edgetpu-compiler.git
# docker run -it --rm -v $(pwd):/home/edgetpu edgetpu_compiler edgetpu_compiler YOUR_MODEL_FILE.tflite
# Shell command for onnx:
# python -m tf2onnx.convert --saved-model tensorflow-model-path --output model.onnx
# Shell command for onnx frozen graph:
# python -m tf2onnx.convert --graphdef model.pb --output model.onnx --opset 11 --inputs=<input_nodes> --outputs=<output_nodes>
# Python command for frozen graph to uff:
# uff.from_tensorflow_frozen_model('model.pb', output_filename='model.uff', output_nodes=[<output_nodes>])


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert model')
    subparsers = parser.add_subparsers(dest='mode')
    frozen_rknn_parser = subparsers.add_parser('frozen_rknn', help='Convert frozen graph to rknn')
    read_frozen_parser = subparsers.add_parser('read_frozen', help='Read nodes from frozen graph')
    frozenfloat_parser = subparsers.add_parser('frozen_float', help='Convert int8 input nodes to fp32 to be compatible with onnx on TensorRT')
    tflite_rknn_parser = subparsers.add_parser('tflite_rknn', help='Convert TFLite model to rknn')
    onnx_rknn_parser = subparsers.add_parser('onnx_rknn', help='Convert onnx model to rknn')
    tf_tftrt_parser = subparsers.add_parser('tf_tftrt', help='Optimize TF saved model with TF-TRT')
    frozen_tftrt_parser = subparsers.add_parser('frozen_tftrt', help='Convert frozen graph to TF-TRT')
    tf_tflite_parser = subparsers.add_parser('tf_tflite', help='Convert TF model to TFLite')
    frozen_savedmodel_parser = subparsers.add_parser('frozen_savedmodel', help='Convert TF1 frozen graph to saved model')
    quant_onnx_parser = subparsers.add_parser('correct_quant_onnx', help='Correct quantized onnx model with unsupported nodes from TF')
    
    
    frozen_rknn_parser.add_argument('-d', '--input_dims', required=True, type=int, help='Model input size')
    frozen_rknn_parser.add_argument('-in', '--input_nodes', required=True, help='Names of input nodes separated by comma')
    frozen_rknn_parser.add_argument('-on', '--output_nodes', required=True, help='Names of output nodes separated by comma')
    frozen_rknn_parser.add_argument('-o', '--output_path', default='model.rknn', help='Path to output model')
    frozen_rknn_parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size of the model, defaults to 1')

    tflite_rknn_parser.add_argument('-d', '--input_dims', required=True, type=int, help='Model input size')
    tflite_rknn_parser.add_argument('-o', '--output_path', default='model_lite.rknn', help='Path to output model')

    onnx_rknn_parser.add_argument('-d', '--input_dims', required=True, type=int, help='Model input size')
    onnx_rknn_parser.add_argument('-o', '--output_path', default='model.rknn', help='Path to output model')

    tf_tftrt_parser.add_argument('-d', '--input_dims', type=int, help='Input dimensions of the model for building TensorRT engines')
    tf_tftrt_parser.add_argument('-p', '--precision_mode', required=True, default='fp32', help='Precision type: \'INT8\', \'FP16\', \'FP32\'')
    tf_tftrt_parser.add_argument('-t', '--input_type', required=True, help='Input type: \'INT8\', \'FP16\', \'FP32\'')
    tf_tftrt_parser.add_argument('-q', '--quantization', help='Do quantization with specified coco dataset')
    tf_tftrt_parser.add_argument('-o', '--output_path', default='saved_model_rt', help='Path to output model')

    frozen_tftrt_parser.add_argument('-o', '--output_path', default='frozen_graph.pb', help='Path to output model')

    frozenfloat_parser.add_argument('-in', '--input_node', required=True, help='Name of input node')
    frozenfloat_parser.add_argument('-d', '--input_dims', required=True, type=int, help='Input dimensions of the model')
    frozenfloat_parser.add_argument('-o', '--output_path', default='model_fp.pb', help='Path to output model')
    frozenfloat_parser.add_argument('-b', '--batch_size', type=int, default=-1, help='Batch size of the model, defaults to -1')

    tf_tflite_parser.add_argument('-q', '--quantization', help='Do quantization with specified coco dataset')
    tf_tflite_parser.add_argument('-d', '--input_dims', type=int, help='Input dimensions to resize dataset images to')
    tf_tflite_parser.add_argument('-i', '--full_integer', action='store_true', default=False, help='Do full integer quantization')
    tf_tflite_parser.add_argument('-o', '--output_path', default='model.tflite', help='Path to output model')

    frozen_savedmodel_parser.add_argument('-in', '--input_node', required=True, help='Input node to model')
    frozen_savedmodel_parser.add_argument('-on', '--output_nodes', required=True, help='Output nodes separated by comma')
    frozen_savedmodel_parser.add_argument('-o', '--output_path', default='saved_model', help='Output path to saved model')

    quant_onnx_parser.add_argument('-o', '--output_path', required=True, help='Output path')
    
    parser.add_argument('model_path', help='Path to input model')

    args = parser.parse_args()
    
    
    # Convert specified model
    if args.mode == 'tflite_rknn':
        from utils.convert.rknn import convert_tflite_rknn
        convert_tflite_rknn(args.model_path, args.output_path, args.input_dims)

    elif args.mode == 'onnx_rknn':
        from utils.convert.rknn import convert_onnx_rknn
        convert_onnx_rknn(args.model_path, args.output_path, args.input_dims)

    elif args.mode == 'frozen_rknn':
        from utils.convert.rknn import convert_frozen_rknn
        convert_frozen_rknn(args.model_path, args.output_path, args.input_dims, args.input_nodes.split(','), args.output_nodes.split(','), args.batch_size)

    elif args.mode == 'frozen_savedmodel':
        from utils.convert.frozen_savedmodel import convert_frozen_savedmodel

        convert_frozen_savedmodel(args.model_path, args.output_path, args.input_node, args.output_nodes.split(','))

    elif args.mode == 'tf_tftrt':
        from utils.convert.tf_tftrt import convert_tf_tftrt

        convert_tf_tftrt(args.model_path, args.output_path, args.input_type, args.precision_mode, args.input_dims, args.quantization)

    elif args.mode == 'frozen_tftrt':
        from utils.convert.frozen_tftrt import convert_frozen_tftrt

        convert_frozen_tftrt(args.model_path, args.output_path)

    elif args.mode == 'tf_tflite':
        from utils.convert.tf_tflite import convert_tf_tflite
        convert_tf_tflite(args.model_path, args.output_path, args.quantization, args.input_dims, args.full_integer)

    elif args.mode == 'read_frozen':
        #import tensorflow as tf
        from utils.convert.frozen_utils import read_frozen

        read_frozen(args.model_path)

    elif args.mode == 'frozen_float':
        from utils.convert.frozen_utils import convert_frozen_float

        convert_frozen_float(args.model_path, args.output_path, args.input_dims, args.input_node, args.batch_size)

    elif args.mode == 'correct_quant_onnx':
        from utils.convert.quant_onnx import quant_onnx

        quant_onnx(args.model_path, args.output_path)

    else:
        print("Conversion not supported")
        exit(1)
