#!/bin/bash
#SBATCH -J script_art633
#SBATCH -N 1
#SBATCH -n 9
#SBATCH --ntasks-per-node=9
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=g.g.a.fouilhe@gmail.com

#parmi les modules à charger
module load cuda/11.8
module load conda
module load gcc/12.3.0
conda activate in-context-learning

python /users/p22034/fouilhe/in-context-learning/src/train.py --config /users/p22034/fouilhe/in-context-learning/src/conf/polynomial_regression.yaml
# python /users/p22034/fouilhe/in-context-learning/src/test.py



