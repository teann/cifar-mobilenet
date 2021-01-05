#!/usr/bin/env bash

#SBATCH --partition=ppc
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
module load anaconda/wml/1.6.2
bootstrap_conda
conda activate wmlce

python network/efficientnetb0.py
