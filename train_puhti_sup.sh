#!/bin/bash
#SBATCH --job-name=PHONG
#SBATCH --account=Project_2001055
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH -o train_o_sup.txt
#SBATCH -e train_e_sup.txt
module load pytorch/1.10

python train.py --root_dir /scratch/project_2001055/dataset/DTU/dtu --num_epochs 32 \
                --batch_size 4 --depth_interval 2.65 --n_depths 8 32 48 --interval_ratios 1.0 2.0 4.0 \
                --optimizer adam --lr 1e-3 --lr_scheduler cosine --num_gpus 4 --loss_type sup --exp_name sup \
                --ckpt_dir /scratch/project_2001055/NovelDepth_models/ckpts --log_dir /scratch/project_2001055/NovelDepth_models/logs

