import tensorrt as trt
from time import time
import numpy as np
from utils.img_utils import load_preproc_images
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
EXPLICIT_PRECISION = 0 #1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def GiB(val):
    return val * 1 << 30


def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return np.expand_dims(np.array([out.host for out in outputs]), 0)


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_PRECISION|EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = GiB(1)
        # Load the Onnx model and parse it in order to populate the TensorRT netw    ork.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        return builder.build_cuda_engine(network)


def inference_onnx_rt(model_path, datatype, input_dims, dataset, batch_size, repeat):
    init_time_start = time()
    with build_engine_onnx(model_path) as engine:
        timestamps, results = list(), list()
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        init_time_end = time()

        timestamps.append(("init_start", init_time_start))
        timestamps.append(("init_end", init_time_end))

        print('Allocation took %f seconds' % (init_time_end - init_time_start))

        with engine.create_execution_context() as context:
           # Inference loop
            total_time, img_cnt = 0.0, 0
            while img_cnt + batch_size <= len(dataset):
                # Load batch
                img_paths = dataset[img_cnt:img_cnt + batch_size]
                img_cnt += batch_size
            
                img_dims, input_tensor = load_preproc_images(img_paths, datatype, input_dims)#.astype(trt.nptype(dtype))
                inputs[0].host = input_tensor
            
                # Inference and append output to results
                inf_start_time = time()
                for i in range(repeat):
                    out = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                inf_end_time = time()
                timestamps.append(("inf_start_batch_%d" % (img_cnt), inf_start_time))
                timestamps.append(("inf_end_batch_%d" % (img_cnt), inf_end_time))
            
                # Log
                duration = inf_end_time-inf_start_time
                total_time += duration
                print('Inference took %f seconds' % (duration))
            
                #print(locations, classes, scores)
                results.append((out, img_dims))
            
            print('Inferenced %d images in %f seconds' % (img_cnt, total_time))
        
            return results, timestamps
