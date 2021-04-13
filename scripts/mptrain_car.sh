python3 ./tools/train_net.py \
  --cfg ./configs/config_car_mono_depth_map_rpn.py \
  --savemodel ./outputs/dsgn_car_mono_depth_map_rpn \
  --start_epoch 1 \
  --lr_scale 50 \
  --epochs 60 \
  -btrain 4 \
  -d 6-7 \
  --multiprocessing-distributed 