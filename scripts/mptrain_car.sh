python3 ./tools/train_net.py \
  --cfg ./configs/config_car_mono.py \
  --savemodel ./outputs/dsgn_car_mono_rpn \
  --start_epoch 1 \
  --lr_scale 50 \
  --epochs 60 \
  -btrain 4 \
  -d 5-6 \
  --multiprocessing-distributed 