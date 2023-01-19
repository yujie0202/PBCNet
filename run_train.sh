#!/bin/bash

device=2
encoder_type="Bind"
batch_size=60
loss_function="mse"
continue_learning=0
hidden_dim=96
p_dropout=0.1
result_file="_ablation_deltadelta_20221207"
# train_path="/home/yujie/leadopt/data/train_files_uni/mean_600000train_all_pair_withoutFEP.csv"
# val_path="/home/yujie/leadopt/data/train_files_uni/train_1_pair.csv"
fold="mse_lr5-7_p0.1_2"
log_frequnency=1000
init_lr=0.0000005
max_lr=0.0000005
final_lr=0.0000005

seed=1
degree_information=0
radius=3
readout_type="AttFP"

cs=0
GCN=1
two_task=0


retrain=0
retrain_model_path="/home/yujie/leadopt/results7/EAT_atomcrossatt_pair_mse_withoutcs_degree2_and_fnn_lr_edge_200/model.pth"


# rcs_sub3_angle_and_dist*0.0005_3


python ./training.py \
--device ${device} \
--encoder_type ${encoder_type} \
--batch_size ${batch_size} \
--loss_function ${loss_function} \
--continue_learning ${continue_learning} \
--hidden_dim  ${hidden_dim} \
--p_dropout ${p_dropout} \
--result_file ${result_file} \
--log_frequency ${log_frequnency} \
--max_lr ${max_lr} \
--init_lr ${init_lr} \
--final_lr ${final_lr} \
--retrain ${retrain} \
--retrain_model_path ${retrain_model_path} \
--degree_information ${degree_information} \
--cs ${cs} \
--fold ${fold} \
--seed ${seed} \
--radius ${radius} \
--readout_type ${readout_type} \
--GCN_ ${GCN} \
--two_task ${two_task} \

