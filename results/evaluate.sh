#!/bin/sh

#Tinker Edge R:
python serial_reader.py eval -d results/inference_classification/tinker/tinker_all.txt results/inference_classification/tinker/pow_*
python serial_reader.py eval -d results/inference_classification/tinker/tinker_all.txt results/inference_classification/tinker/ts_*

#Jetson Nano:
python serial_reader.py eval -d results/inference_classification/jetson/jetson_all.txt results/inference_classification/jetson/pow_*
python serial_reader.py eval -d results/inference_classification/jetson/jetson_all.txt results/inference_classification/jetson/ts_*

#Jetson Nano CPU:
python serial_reader.py eval -d results/inference_classification/jetson_cpu/jetson_cpu_all.txt results/inference_classification/jetson_cpu/pow_*
python serial_reader.py eval -d results/inference_classification/jetson_cpu/jetson_cpu_all.txt results/inference_classification/jetson_cpu/ts_*

#Coral Dev Board:
python serial_reader.py eval -d results/inference_classification/coral/coral_all.txt results/inference_classification/coral/pow_*
python serial_reader.py eval -d results/inference_classification/coral/coral_all.txt results/inference_classification/coral/ts_*

#Raspberry Pi:
python serial_reader.py eval -d results/inference_classification/rpi/rpi_all.txt results/inference_classification/rpi/pow_*
python serial_reader.py eval -d results/inference_classification/rpi/rpi_all.txt results/inference_classification/rpi/ts_*

#Arduino:
python serial_reader.py eval -d results/inference_classification/arduino/ts_arduino_mn1.txt
python serial_reader.py eval_power results/inference_classification/arduino/pow_ts_arduino_init.txt
python serial_reader.py eval_power results/inference_classification/arduino/pow_ts_arduino_inf.txt

#Idle states:
python serial_reader.py eval_power results/idle/*
python serial_reader.py eval_power results/idle_nolan/*
