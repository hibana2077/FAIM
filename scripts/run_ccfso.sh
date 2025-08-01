#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1            
#PBS -l ncpus=16            
#PBS -l mem=20GB           
#PBS -l walltime=00:60:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

nvidia-smi >> gpu-info.txt

cd ..

source /scratch/rp06/sl5952/CC-FSO/.venv/bin/activate
python3 train.py --dataset cotton80 --model resnet152 --loss ce --optimizer riemannian_sgd --epochs 100 --batch_size 32 --lr 0.001 >> out_a100.txt

