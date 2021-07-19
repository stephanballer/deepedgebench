(python mem_reader.py ../mem_coco.txt) &
(python test.py tensorflow_1 models/ssd_mobilenet_v2_coco_2018_03_29/saved_model datasets/instances_val2017.json -d 300 -n 1000 -f ../inf_coco.txt) &

for job in `jobs -p`; do
    wait $job
done
