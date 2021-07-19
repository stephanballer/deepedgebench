import tensorflow as tf
from utils.img_utils import load_preproc_images

def convert_tf_tflite(model_path, output_path, dataset=None, input_dims=None, full_integer=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.allow_custom_ops = True

    if dataset is not None:
        from pycocotools.coco import COCO

        if input_dims is not None:
            input_dims = (input_dims, input_dims)

        def dataset_fun():
            coco_gt = COCO(dataset)
            imgs_coco = coco_gt.loadImgs(coco_gt.getImgIds())[:200]
            for img in imgs_coco:
                sample = load_preproc_images([img], 'fp32', input_dims)[1]
                yield [sample]

        converter.representative_dataset = dataset_fun
        if full_integer:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

    model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(model)
