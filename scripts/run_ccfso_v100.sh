#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1            
#PBS -l ncpus=12            
#PBS -l mem=20GB           
#PBS -l walltime=00:50:00  
#PBS -l wd                  
#PBS -l storage=scratch/rp06

module load cuda/12.6.2
# module load python3/3.10.4

nvidia-smi >> gpu-info.txt

cd ..

source /scratch/rp06/sl5952/CC-FSO/.venv/bin/activate
# python3 train.py --dataset cotton80 --model resnet50 --loss ce --optimizer riemannian_sgd --epochs 100 --batch_size 32 --lr 0.001 --feature_dim 2048 >> out_v100.txt
# python3 train.py --dataset cotton80 --model vit_small_patch16_384.augreg_in21k_ft_in1k --loss ce --optimizer adamw --epochs 100 --batch_size 32 --lr 0.0001 --image_size 384 --feature_dim 2048 >> out_v100.txt
python3 train.py --dataset cotton80 --model tiny_vit_21m_384 --loss ce --optimizer adamw --epochs 100 --batch_size 32 --lr 0.0001 --image_size 384 --feature_dim 768 >> out_v100.txt
