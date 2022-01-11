#!/bin/bash

python train.py --root_dir /scratch/project_2001055/dataset/DTU/dtu --num_epochs 32 \
                --batch_size 4 --depth_interval 2.65 --n_depths 8 32 48 --interval_ratios 1.0 2.0 4.0 \
                --optimizer adam --lr 1e-3 --lr_scheduler cosine --num_gpus 4 --loss_type unsup --exp_name unsup \
                --ckpt_dir /scratch/project_2001055/NovelDepth_models/ckpts --log_dir /scratch/project_2001055/NovelDepth_models/logs

