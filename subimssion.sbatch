#!/bin/sh
# fichier submission.SBATCH
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --job-name="Synthetic_supervised_learning"
#SBATCH --output=%x.out
#SBATCH --time=120:00
#SBATCH --error=%x.out
#SBATCH --mail-user=tbertrand@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,FAIL,END

python3 supervised_learning_ULM.py                                                                                                        
