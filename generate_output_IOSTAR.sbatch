#!/bin/sh
# fichier submission.SBATCH
#SBATCH --nodes=1
#SBATCH -c 10
#SBATCH --job-name="IOSTAR_output_geodesic_tree"
#SBATCH --output=%x.out
#SBATCH --time=96:00:00
#SBATCH --error=%x.out
#SBATCH --mail-user=tbertrand@ceremade.dauphine.fr
#SBATCH --mail-type=BEGIN,FAIL,END

python3 ./figures_IOSTAR.py --nb 0
python3 ./figures_IOSTAR.py --nb 1
python3 ./figures_IOSTAR.py --nb 2
python3 ./figures_IOSTAR.py --nb 3
python3 ./figures_IOSTAR.py --nb 4
python3 ./figures_IOSTAR.py --nb 5
python3 ./figures_IOSTAR.py --nb 6
python3 ./figures_IOSTAR.py --nb 7
python3 ./figures_IOSTAR.py --nb 8
python3 ./figures_IOSTAR.py --nb 9
python3 ./figures_IOSTAR.py --nb 10
python3 ./figures_IOSTAR.py --nb 11
python3 ./figures_IOSTAR.py --nb 12
python3 ./figures_IOSTAR.py --nb 13
python3 ./figures_IOSTAR.py --nb 14
python3 ./figures_IOSTAR.py --nb 15
python3 ./figures_IOSTAR.py --nb 16
python3 ./figures_IOSTAR.py --nb 17
python3 ./figures_IOSTAR.py --nb 18
python3 ./figures_IOSTAR.py --nb 19
python3 ./figures_IOSTAR.py --nb 20
