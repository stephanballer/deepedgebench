from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os
import tensorflow as tf
import numpy as np

def convert_tf_tftrt(model_path, output_path, input_type, precision_mode, input_dims=None, dataset=None):
        # Shell command:
        #  saved_model_cli convert --dir ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model --output_dir ssd_mobilenet_v2_320x320_coco17_tpu-8_rt --tag_set serve tensorrt --precision_mode FP16
        
        # Set conversion parameters
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
        conversion_params = conversion_params._replace(
                    max_workspace_size_bytes=(1<<32))
        conversion_params = conversion_params._replace(precision_mode=precision_mode)
        conversion_params = conversion_params._replace(use_calibration=dataset is not None)
        conversion_params = conversion_params._replace(
                    maximum_cached_engines=100)

        if dataset is not None:
            dataset_dir = os.path.dirname(os.path.realpath(dataset))
    
            with open(dataset, 'r') as f:
                dataset = list()
                for line in f.readlines():
                    filename, gt = line.split()
                    dataset.append(os.path.join(dataset_dir, filename))
    
                    if len(dataset) >= 200:
                        break

#            from pycocotools.coco import COCO
 
            dims = None
            if input_dims is not None:
                dims = (input_dims, input_dims)

            def dataset_fun():
                for img in dataset:
                    sample = load_preproc_images([img], input_type, dims)[1]
                    yield (sample,)
#                coco_gt = COCO(dataset)
#                imgs_coco = coco_gt.loadImgs(coco_gt.getImgIds())[:200]
#                for img in imgs_coco:
#                    sample = load_preproc_images([img], input_type, dims)[1]
#                    yield (sample,)
        else:
            dataset_fun = None

       
        if input_type.lower() == 'int8':
            inp_type = np.uint8
            #def my_calibration_input_fn():
            #    inp1 = np.random.normal(size=(1, args.input_dims, args.input_dims, 3)).astype(inp_type)
            #    yield(inp1,)
        elif input_type.lower() == 'fp16':
            inp_type = np.float16
        elif input_type.lower() == 'fp32':
            inp_type = np.float32
        
        # Convert and save model
        print("Converting model...")
        converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_path, conversion_params=conversion_params)
        converter.convert(calibration_input_fn=dataset_fun)
        if input_dims is not None:
            def my_input_fn():
                inp1 = np.random.normal(size=(1, input_dims, input_dims, 3)).astype(inp_type)
                yield(inp1,)
        else:
            my_input_fn = None
        converter.build(input_fn=my_input_fn)
        converter.save(output_path)
 

def convert_frozen_tftrt(model_path, output_path):
    with tf.Session() as sess:
        # First deserialize your frozen graph:
        with tf.gfile.GFile(model_path, 'rb') as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())
        # Now you can create a TensorRT inference graph from your
        # frozen graph:
        converter = trt.TrtGraphConverter(
    	    input_graph_def=frozen_graph,
    	    nodes_blacklist=['logits', 'classes']) #output nodes
        trt_graph = converter.convert()
        # Import the TensorRT graph into a new graph and run:
        output_node = tf.import_graph_def(
            trt_graph,
            return_elements=['logits', 'classes'])

        tf.io.write_graph(graph_or_graph_def=trt_graph,
                      name=output_path,
                      as_text=False)
