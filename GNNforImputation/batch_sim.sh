#!/bin/bash

for c in {1..2}
do
for clean_file in data/sim3000/simulation3_case2_5_7_50_noD_noE.csv
#for clean_file in data/sim3000/simulation1_case2_rnd.csv
do
  dirty_file=$(basename -s .csv $clean_file)_B_50.csv
  python main_multilabel.py \
  --ground_truth $clean_file --dirty_dataset data/sim3000/sim3_dirty/$dirty_file \
  --epochs 200 --predictor_layers 4 --head_layers 4 --graph_layers 2 --h_feats 512 --dropout 0.3 --gnn_feats 128 \
  --architecture multitask --weight_decay 1e-4 --learning_rate 0.001 --aggr gcn --loss focal --loss_gamma 2 \
  --grace 250
done
done