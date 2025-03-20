@echo off

REM Task 1
python train_wandb.py -epoch 15 -lr 0.01
python train_wandb.py -epoch 15 -lr 0.05
python train_wandb.py -epoch 15 -lr 0.2

REM Task 2
python train_wandb.py -epoch 300 -lr 0.05 -scheduler False
python train_wandb.py -epoch 300 -lr 0.05 -scheduler True

REM Task 3
python train_wandb.py -epoch 300 -lr 0.05 -scheduler True -use_weight_decay True -weight_decay 1e-04
python train_wandb.py -epoch 300 -lr 0.05 -scheduler True -use_weight_decay True -weight_decay 5e-04

REM Task 4
python train_wandb.py -epoch 300 -lr 0.05 -scheduler True -use_weight_decay True -weight_decay 5e-04 -record_grad True
python train_wandb.py -epoch 300 -lr 0.05 -scheduler True -use_weight_decay True -weight_decay 5e-04 -sigma_block_ind 4,5,6,7,8,9,10 -record_grad True

echo All tasks completed!
pause
