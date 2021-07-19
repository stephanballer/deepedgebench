from rknn.api import RKNN  
 
def convert_rknn(model_path, output_path, input_size, input_nodes=None, output_nodes=None, quant_dataset_path=None, platform='tf', batch_size=1):
    # Initialize and configure converter
    rknn = RKNN()
    #ret = rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2', target_platform='rk3399pro') # [0,1]
    #ret = rknn.config(channel_mean_value='127.5 127.5 127.5 127.5', reorder_channel='0 1 2', target_platform='rk3399pro') #[-1,1]
    ret = rknn.config(reorder_channel='0 1 2', target_platform='rk3399pro')
    if ret != 0:
        print("Error configuring RKNN")
        exit()
    

    # Load model
    if platform == 'tflite':
        ret = rknn.load_tflite(model=model_path)
    elif platform == 'tf':
        ret = rknn.load_tensorflow(tf_pb=model_path,
                inputs=input_nodes,
                outputs=output_nodes,
                input_size_list=[[input_size, input_size, 3] for _ in range(batch_size)])
    elif platform == 'onnx':
        ret = rknn.load_onnx(model=model_path)
    else:
        exit(1)
    
    
    if ret != 0:
        print("Error loading model")
        exit()
    
    # Convert model
    if quant_dataset_path is None:
        ret = rknn.build(do_quantization=False)
    else: 
        ret = rknn.build(do_quantization=True, dataset=quant_dataset_path)

    if ret != 0:
        print("Error building model")
        exit()
    
    # Save model
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print("Error building model")
    
    # Cleanup
    rknn.release()

def convert_onnx_rknn(model_path, output_path, input_size, quant_dataset_path=None):
    convert_rknn(model_path, output_path, input_size, None, None, quant_dataset_path, 'onnx')

def convert_tflite_rknn(model_path, output_path, input_size, quant_dataset_path=None):
    convert_rknn(model_path, output_path, input_size, None, None, quant_dataset_path, 'tflite')

def convert_frozen_rknn(model_path, output_path, input_size, input_nodes, output_nodes, batch_size, quant_dataset_path=None):
    convert_rknn(model_path, output_path, input_size, input_nodes, output_nodes, quant_dataset_path, 'tf', batch_size)
