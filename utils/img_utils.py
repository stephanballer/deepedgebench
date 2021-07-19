import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import os.path
import numpy as np


# Draws a list of bounding boxes and labels to image and cycle through
# given colors
def draw_bounding_boxes(img, bboxs, labels, colors):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for i, (bbox, label) in enumerate(zip(bboxs, labels)):
        draw.rectangle(bbox, outline=colors[i % len(colors)])
        text_width, text_height = font.getsize(label)
        draw.rectangle((bbox[0], bbox[3]-text_height, bbox[0]+text_width, bbox[3]), fill=colors[i % len(colors)],
                outline=colors[i % len(colors)])
        draw.text((bbox[0], bbox[3]-text_height), label, fill=(0,0,0), font=font)

    return img


# Annotate and save image with multiple bbox color schemes (to compare datasets)
def save_annot_image(image_path, output_path, bbox_list, cat_list, col_list):
    img = load_image(image_path)

    for bboxs, cats, cols in zip(bbox_list, cat_list, col_list):
        img = draw_bounding_boxes(img, bboxs, cats, cols)

    img.save(output_path)


# Load image from file or url
def load_image(image_path):
    if not os.path.isfile(image_path):
        response = requests.get(image_path)
        image_path = BytesIO(response.content)
    
    return Image.open(image_path)


# Load images from path/url list and adjust them to model input. Returns
# array of requested images and their initial dimensions
def load_preproc_images(image_paths, datatype, dims=None):
    img_dims, img_tensors = list(), list()

    # Load images and resize
    for image_path in image_paths:
        img = load_image(image_path)

        img_dims.append(img.size)
        if dims is not None:
            img = img.resize(dims, Image.ANTIALIAS)

        # Get input type and transform input to match type
        type_str = datatype.lower()
        if type_str == 'int8':
            img_tensor = np.array(img, dtype=np.uint8)
        elif type_str == 'fp16':
            img_tensor = ((np.array(img, dtype=np.float16)-127.5)/127.5)
        elif type_str == 'fp32':
            img_tensor = ((np.array(img, dtype=np.float32)-127.5)/127.5)
        else:
            print('Error: Input type not supported!')
            exit(1)

        # Convert grayscale and alpha images to rgb image
        if len(img_tensor.shape) == 2:
            img_tensor = np.repeat(img_tensor[:,:,np.newaxis], 3, axis=2)
        elif img_tensor.shape[2] == 4:
            img_tensor = img_tensor[:,:,:3]

        img_tensors.append(img_tensor)

    return img_dims, np.array(img_tensors)
