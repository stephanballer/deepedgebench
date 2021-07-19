import tensorflow as tf
import graphsurgeon as gs

def convert_frozen_float(model_path, output_path, input_dims, input_node, batch_size=-1):
    graph = gs.DynamicGraph(model_path)
    image_tensor = graph.find_nodes_by_name(input_node)
    
    print('Found Input: ', image_tensor)
    
    cast_nodes = graph.find_nodes_by_name('Cast') #Replace Cast with ToFloat if using tensorflow <1.15
    cast_nodes += graph.find_nodes_by_name('ToFloat') #Replace Cast with ToFloat if using tensorflow <1.15
    
    for cast_node in cast_nodes:
        print('Old field', cast_node.attr['SrcT'])
        cast_node.attr['SrcT'].type=1 #Changing Expected type to float
        print('New field', cast_node.attr['SrcT'])
    
    graph_input = gs.create_plugin_node(name=input_node, op='Placeholder', shape=(batch_size, input_dims, input_dims, 3), dtype=tf.float32)
    namespace_plugin_map = {input_node: graph_input}
    graph.collapse_namespaces(namespace_plugin_map)
    graph.write(output_path)

def read_frozen(model_path):
    graph = gs.DynamicGraph(model_path)
    print(graph.graph_inputs)
    print(graph.graph_outputs)

        
    def load_model():
        with tf.compat.v1.gfile.GFile(path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
    
        with tf.compat.v1.Graph().as_default() as graph:
            tf.compat.v1.import_graph_def(graph_def, name="")
        return graph
    
    path = model_path
    graph = load_model()
    with tf.compat.v1.Session(graph=graph) as sess:
        for op in graph.get_operations():
            print(op.name)

        flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
        print('Model needs %s FLOPS after freezing' % (flops.total_float_ops))
