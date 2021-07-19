import tensorflow.compat.v1 as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

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
