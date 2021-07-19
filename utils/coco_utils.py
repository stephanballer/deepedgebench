from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from utils.img_utils import save_annot_image
import numpy as np
import json

def write_coco_results(results, file_path, dataset, min_score):
    # Create result annotation file for Coco
    coco_results = list()
    img_id_cnt = 0
    # Iterate over every inference batch
    for result, img_dims in results:
        # Iterate over every image in batch
        for i, (width, height) in enumerate(img_dims):
            # Adjust normalized bboxes to absolute pixel coordiantes
            bboxs = (result[0][i] * np.array([height, width, height, width])).tolist()
            img_id = dataset[img_id_cnt]['id']
            cats = result[1][i]

            # Print coco result file
            for (cat, bbox, score) in zip(cats, bboxs, result[2][i]):
                if score >= min_score:
                    coco_results.append(
                        {
                            'image_id': img_id,
                            'category_id': int(cat),
                            'bbox': [  # Format bbox to x, y, width, height
                                bbox[1],
                                bbox[0],
                                (bbox[3] - bbox[1]),
                                (bbox[2] - bbox[0])
                            ],
                            'score': float(score)
                        }
                    )
    
            img_id_cnt += 1

    if len(dataset) > 0:
        with open(file_path, 'w') as f:
            json.dump(coco_results, f) 
            

def coco_eval(coco_gt, coco_det, dataset):
    # Calculate Coco AP and AR metrics
    coco_eval = COCOeval(coco_gt, coco_det, 'bbox')
    # Only consider inferenced images
    coco_eval.params.imgIds = [img['id'] for img in dataset]
    coco_eval.evaluate()
    coco_eval.accumulate()
    # Print summary
    coco_eval.summarize()
    return coco_eval.stats

# Annotate images with boxes and categories from dataset and save them to
# image folder
def save_coco_imgs(coco, images, output_path, colors):
    # Load boxes and categories from image ids
    for img in images:
        img_id = img['id']
        ann_ids = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        bboxs = [[
            ann['bbox'][0],
            ann['bbox'][1],
            ann['bbox'][0] + ann['bbox'][2],
            ann['bbox'][1] + ann['bbox'][3]] for ann in ann_ids]
        cat_names = [cat['name'] for cat in coco.loadCats(
            [ann['category_id'] for ann in ann_ids])]

        # Load image, draw boxes and save them for comparison
        save_annot_image(img['coco_url'],'%s/%d.png' % (output_path, img_id),
                [bboxs], [cat_names], [colors])


# Annotate images with boxes and categories from two datasets and save them to
# image folder
def save_compare_coco_imgs(cocos, images, output_path, color_schemes):
    # Load boxes and categories from image ids
    for img in images:
        img_id = img['id']

        bboxs_list, cats_list = list(), list()
        for i, coco in enumerate(cocos):
            ann_ids = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
            bboxs_list.append([[
                ann['bbox'][0],
                ann['bbox'][1],
                ann['bbox'][0] + ann['bbox'][2],
                ann['bbox'][1] + ann['bbox'][3]] for ann in ann_ids])

            cats_list.append(["ds_%d: " % (i) + cat['name'] for cat in coco.loadCats(
                [ann['category_id'] for ann in ann_ids])])

        # Load image, draw boxes and save them for comparison
        save_annot_image(img['coco_url'], '%s/cmp_%d.png' % (output_path, img_id),
                bboxs_list, cats_list, color_schemes)
