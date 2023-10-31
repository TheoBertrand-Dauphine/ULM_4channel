#!/bin/sh
# fichier submission.SBATCH
#SBATCH --nodes=1
#SBATCH -c 5
#SBATCH --job-name="IOSTAR_output_tuning"
#SBATCH --output=%x.out
#SBATCH --time=96:00:00
#SBATCH --error=%x.out
#SBATCH --mail-user=tbertrand@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,FAIL,END

python3 -m wandb agent tbertrand/ULM_4channel/ll1r13hu --count 10
