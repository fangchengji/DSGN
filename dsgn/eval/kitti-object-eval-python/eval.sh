#!/bin/bash
echo $1
if [ ! -n "$2" ] ; then
    class="0"
else
    class=$2
fi
echo $class
python3 evaluate.py evaluate \
    --label_path=/chj/jifangcheng/datasets/kitti/training/label_2/ \
    --result_path=$1 \
    --current_class=$class --coco=False


