#!/usr/bin/env bash

#SBATCH --partition=ppc
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
module load anaconda/wml/1.6.2
bootstrap_conda
conda activate wmlce

python train.py --net efficientnetb0 --batch_size 64
# python -c "import tf_slim as slim; eval = slim.evaluation.evaluate_once"
