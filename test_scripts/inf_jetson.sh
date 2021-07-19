#!/bin/bash

export PYTHONPATH=$(pwd)

DIMS=224
NUM_INF=5
NUM_REP=1000
MODEL=mobilenet_v2
MODEL_SHORT=mn2
DEVICE=jetson


# V2 tf
sleep 5
python test.py tensorflow_1 models/$MODEL/saved_model ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../pow_ts_"$DEVICE"_tf_"$MODEL_SHORT".txt -r $NUM_REP

# V2 tflite
sleep 5
python test.py tflite models/$MODEL/model.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../pow_ts_"$DEVICE"_tflite_"$MODEL_SHORT".txt -r $NUM_REP

# V2 tflite quant
sleep 5
python test.py tflite models/$MODEL/model_quant.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -f ../pow_ts_"$DEVICE"_tflite_"$MODEL_SHORT"q.txt -r $NUM_REP

# V2 tf_trt
sleep 5
python test.py tensorflow_1 models/$MODEL/saved_model_rt ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../pow_ts_"$DEVICE"_tftrt_mn2.txt -r $NUM_REP

# V2 onnx_trt
sleep 5
python test.py onnx_trt models/$MODEL/model.onnx ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -f ../pow_ts_"$DEVICE"_onnxtrt_mn2.txt -r $NUM_REP -t fp32

MODEL=mobilenet_v1
MODEL_SHORT=mn1
DIMS=128

# V1 tflite quant
sleep 5
python test.py tflite models/$MODEL/model_quant.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -f ../pow_ts_"$DEVICE"_tflite_mn1q.txt -r $NUM_REP



NUM_INF=5000
NUM_REP=1

MODEL=mobilenet_v2
MODEL_SHORT=mn2
DIMS=224


# V2 tf
sleep 5
python test.py tensorflow_1 models/$MODEL/saved_model ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../ts_"$DEVICE"_tf_"$MODEL_SHORT".txt -r $NUM_REP

# V2 tflite
sleep 5
python test.py tflite models/$MODEL/model.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../ts_"$DEVICE"_tflite_"$MODEL_SHORT".txt -r $NUM_REP

# V2 tflite quant
sleep 5
python test.py tflite models/$MODEL/model_quant.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -f ../ts_"$DEVICE"_tflite_"$MODEL_SHORT"q.txt -r $NUM_REP

# V2 tf_trt
sleep 5
python test.py tensorflow_1 models/$MODEL/saved_model_rt ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../ts_"$DEVICE"_tftrt_mn2.txt -r $NUM_REP

# V2 onnx_trt
sleep 5
python test.py onnx_trt models/$MODEL/model.onnx ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -f ../ts_"$DEVICE"_onnxtrt_mn2.txt -r $NUM_REP -t fp32

MODEL=mobilenet_v1
MODEL_SHORT=mn1
DIMS=128

# V1 tflite quant
sleep 5
python test.py tflite models/$MODEL/model_quant.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -f ../ts_"$DEVICE"_tflite_mn1q.txt -r $NUM_REP


