#!/bin/bash

export PYTHONPATH=$(pwd)

DIMS=224
NUM_INF=5
NUM_REP=1000
MODEL=mobilenet_v2
MODEL_SHORT=mn2
DEVICE=coral


# V2 tf
sleep 5
python test.py tensorflow_frozen -in input -on MobilenetV2/Predictions/Reshape_1:0 models/$MODEL/model.pb ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../pow_ts_"$DEVICE"_tf_"$MODEL_SHORT".txt -r $NUM_REP

# V2 tflite
sleep 5
python test.py tflite models/$MODEL/model.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../pow_ts_"$DEVICE"_tflite_"$MODEL_SHORT".txt -r $NUM_REP

# V2 tflite quant
sleep 5
python test.py tflite models/$MODEL/model_quant.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -f ../pow_ts_"$DEVICE"_tflite_"$MODEL_SHORT"q.txt -r $NUM_REP

# V2 tflite quant tpu
sleep 5
python test.py tflite -u models/$MODEL/model_quant_edgetpu.tflite ../imagenet/ILSVRC2012_dataset.txt -d $DIMS -n $NUM_INF -f ../pow_ts_"$DEVICE"_litetpu_"$MODEL_SHORT"q.txt -r $NUM_REP

MODEL=$MODEL
MODEL_SHORT=mn1
DIMS=$DIMS

# V1 tflite quant
sleep 5
python test.py tflite models/$MODEL/model_quant.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -f ../pow_ts_"$DEVICE"_tflite_mn1q.txt -r $NUM_REP

# V1 tflite quant tpu
sleep 5
python test.py tflite -u models/$MODEL/model_quant_edgetpu.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -f ../pow_ts_"$DEVICE"_litetpu_mn1q.txt -r $NUM_REP

NUM_INF=5000
NUM_REP=1

MODEL=mobilenet_v2
MODEL_SHORT=mn2
DIMS=224


# V2 tf
sleep 5
python test.py tensorflow_frozen -in input -on MobilenetV2/Predictions/Reshape_1:0 models/$MODEL/model.pb ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../ts_"$DEVICE"_tf_"$MODEL_SHORT".txt -r $NUM_REP

# V2 tflite
sleep 5
python test.py tflite models/$MODEL/model.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -t fp32 -f ../ts_"$DEVICE"_tflite_"$MODEL_SHORT".txt -r $NUM_REP

# V2 tflite quant
sleep 5
python test.py tflite models/$MODEL/model_quant.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -f ../ts_"$DEVICE"_tflite_"$MODEL_SHORT"q.txt -r $NUM_REP

# V2 tflite quant tpu
sleep 5
python test.py tflite -u models/$MODEL/model_quant_edgetpu.tflite ../imagenet/ILSVRC2012_dataset.txt -d $DIMS -n $NUM_INF -f ../ts_"$DEVICE"_litetpu_"$MODEL_SHORT"q.txt -r $NUM_REP

MODEL=$MODEL
MODEL_SHORT=mn1
DIMS=$DIMS

# V1 tflite quant
sleep 5
python test.py tflite models/$MODEL/model_quant.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -f ../ts_"$DEVICE"_tflite_mn1q.txt -r $NUM_REP

# V1 tflite quant tpu
sleep 5
python test.py tflite -u models/$MODEL/model_quant_edgetpu.tflite ../imagenet/ILSVRC2012_dataset.txt -n $NUM_INF -d $DIMS -f ../ts_"$DEVICE"_litetpu_mn1q.txt -r $NUM_REP
