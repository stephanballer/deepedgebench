import tensorflow as tf
 
def convert_tf1_tensorrt(model_path, output_path, input_type
    # converter = trt.TrtGraphConverter(
    #     input_saved_model_dir='%s/saved_model' % (args.model_path),
    #     max_workspace_size_bytes=(1<32),
    #     precision_mode=args.input_type,
    #     maximum_cached_engines=100)
    # converter.convert()
    # converter.save('%s/saved_model_rt' % (args.model_path))
    # with tf.compat.v1.Session() as sess:
    #     tf.compat.v1.saved_model.loader.load(sess, export_dir='%s/saved_model' % (args.model_path), tags=[])
    #     builder = tf.compat.v1.saved_model.builder.SavedModelBuilder('%s/saved_model_tmp' % (args.model_path))
    #     builder.add_meta_graph_and_variables(sess, [tf.compat.v1.saved_model.tag_constants.SERVING],)
    #     builder.save()
    
    params = tf.experimental.tensorrt.ConversionParams(
                precision_mode=args.input_type)
    converter = tf.experimental.tensorrt.Converter(
                input_saved_model_dir=model_path, conversion_params=params)
    converter.convert()
    converter.save(output_path)
