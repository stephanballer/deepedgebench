#!/bin/bash

export PYTHONPATH=$(pwd)

DIMS=224
NUM_INF=5
NUM_REP=1000
MODEL=mobilenet_v2
MODEL_SHORT=mn2
DEVICE=jetson_cpu

export CUDA_VISIBLE_DEVICES=

# V2 tf saved model
sleep 5
python test.py tensorflow_1 models/$MODEL/saved_model ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../pow_ts_"$DEVICE"_tf_"$MODEL_SHORT".txt -r $NUM_REP

# V2 tf
sleep 5
python test.py tensorflow_frozen -in input -on MobilenetV2/Predictions/Reshape_1:0 models/mobilenet_v2/model.pb ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../pow_ts_"$DEVICE"_tffg_"$MODEL_SHORT".txt -r $NUM_REP


NUM_INF=5000
NUM_REP=1

# V2 tf saved model
sleep 5
python test.py tensorflow_1 models/$MODEL/saved_model ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../ts_"$DEVICE"_tf_"$MODEL_SHORT".txt -r $NUM_REP

# V2 tf
sleep 5
python test.py tensorflow_frozen -in input -on MobilenetV2/Predictions/Reshape_1:0 models/mobilenet_v2/model.pb ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../ts_"$DEVICE"_tffg_"$MODEL_SHORT".txt -r $NUM_REP

unset CUDA_VISIBLE_DEVICES
