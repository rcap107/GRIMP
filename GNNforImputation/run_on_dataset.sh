#!/bin/bash

dset_name=$1

clean_dataset=data/new_datasets/$dset_name.csv

predictor_layers=2
head_layers=2
dropout=0.7
loss_gamma=1
comb_size=1

for p in 20 60
do
for ext in '' _"$p""_misf" _"$p""_new"
# testing oracle configuration on all columns
do
f=data/new_datasets_dirty/$dset_name'_all_columns_'$p'.csv'
  echo $f
  python main_multilabel.py --ground_truth $clean_dataset \
  --dirty_dataset $f \
  --architecture multitask --graph_layers 2 --gnn_feats 128 --h_feats 128 --predictor_layers $predictor_layers \
  --head_layers $head_layers \
  --training_subset target --loss focal --loss_gamma $loss_gamma --epochs 250 --grace 100 --dropout $dropout \
  --comb_size $comb_size \
  --text_emb data/pretrained/"$dset_name""$ext".emb

f=data/new_datasets_dirty/$dset_name'_all_columns_'$p'.csv'
  echo $f
  python main_multilabel.py --ground_truth $clean_dataset \
  --dirty_dataset $f \
  --architecture multitask --graph_layers 2 --gnn_feats 128 --h_feats 128 --predictor_layers $predictor_layers \
  --head_layers $head_layers \
  --training_subset target --loss focal --loss_gamma $loss_gamma --epochs 250 --grace 100 --dropout $dropout \
  --comb_size $comb_size --ignore_num_cols \
  --text_emb data/pretrained/"$dset_name""$ext".emb
done
done

## testing oracle configuration on non-numeric columns
#for p in 20 60
#do
#f=data/new_datasets_dirty/$dset_name'_all_columns_'$p'.csv'
#  echo $f
#  python main_multilabel.py --ground_truth $clean_dataset \
#  --dirty_dataset $f \
#  --architecture multitask --graph_layers 2 --gnn_feats 128 --h_feats 128 --predictor_layers $predictor_layers \
#  --head_layers $head_layers \
#  --training_subset target --loss focal --loss_gamma $loss_gamma --epochs 250 --grace 100 --dropout $dropout \
#  --comb_size $comb_size --ignore_num_cols \
#  --text_emb data/pretrained/"$dset_name".emb
#done
#
#
## testing misf on all columns
#for p in 20 60
#do
#f=data/new_datasets_dirty/$dset_name'_all_columns_'$p'.csv'
#  echo $f
#  python main_multilabel.py --ground_truth $clean_dataset \
#  --dirty_dataset $f \
#  --architecture multitask --graph_layers 2 --gnn_feats 128 --h_feats 128 --predictor_layers $predictor_layers \
#  --head_layers $head_layers \
#  --training_subset target --loss focal --loss_gamma $loss_gamma --epochs 250 --grace 100 --dropout $dropout \
#  --comb_size $comb_size \
#  --text_emb data/pretrained/"$dset_name"_"$p"_misf.emb
#done
#
#
## testing misf on non-numeric columns
#for p in 20 60
#do
#f=data/new_datasets_dirty/$dset_name'_all_columns_'$p'.csv'
#  echo $f
#  python main_multilabel.py --ground_truth $clean_dataset \
#  --dirty_dataset $f \
#  --architecture multitask --graph_layers 2 --gnn_feats 128 --h_feats 128 --predictor_layers $predictor_layers \
#  --head_layers $head_layers \
#  --training_subset target --loss focal --loss_gamma $loss_gamma --epochs 250 --grace 100 --dropout $dropout \
#  --comb_size $comb_size --ignore_num_cols \
#  --text_emb data/pretrained/"$dset_name"_"$p"_misf.emb
#done
#
## testing embdi+null edges on all columns
#for p in 20 60
#do
#f=data/new_datasets_dirty/$dset_name'_all_columns_'$p'.csv'
#  echo $f
#  python main_multilabel.py --ground_truth $clean_dataset \
#  --dirty_dataset $f \
#  --architecture multitask --graph_layers 2 --gnn_feats 128 --h_feats 128 --predictor_layers $predictor_layers \
#  --head_layers $head_layers \
#  --training_subset target --loss focal --loss_gamma $loss_gamma --epochs 250 --grace 100 --dropout $dropout \
#  --comb_size $comb_size \
#  --text_emb data/pretrained/"$dset_name"_"$p"_new.emb
#done
#
## testing embdi+null edges on non-numeric columns
#for p in 20 60
#do
#f=data/new_datasets_dirty/$dset_name'_all_columns_'$p'.csv'
#  echo $f
#  python main_multilabel.py --ground_truth $clean_dataset \
#  --dirty_dataset $f \
#  --architecture multitask --graph_layers 2 --gnn_feats 128 --h_feats 128 --predictor_layers $predictor_layers \
#  --head_layers $head_layers \
#  --training_subset target --loss focal --loss_gamma $loss_gamma --epochs 250 --grace 100 --dropout $dropout \
#  --comb_size $comb_size --ignore_num_cols \
#  --text_emb data/pretrained/"$dset_name"_"$p"_new.emb
#done

## testing embdi+original edges on all columns
#for p in 20 60
#do
#f=data/new_datasets_dirty/$dset_name'_all_columns_'$p'.csv'
#  echo $f
#  python main_multilabel.py --ground_truth $clean_dataset \
#  --dirty_dataset $f \
#  --architecture multitask --graph_layers 2 --gnn_feats 128 --h_feats 128 --predictor_layers 2 --head_layers 2 \
#  --training_subset target --loss focal --epochs 250 --grace 100 --dropout 0.50 --comb_size 1 --flag_rid --batchnorm \
#  --text_emb data/pretrained/"$dset_name"_"$p"_embdi.emb
#done
#
## testing embdi+original edges on non-numeric columns
#for p in 20 60
#do
#f=data/new_datasets_dirty/$dset_name'_all_columns_'$p'.csv'
#  echo $f
#  python main_multilabel.py --ground_truth $clean_dataset \
#  --dirty_dataset $f \
#  --architecture multitask --graph_layers 2 --gnn_feats 128 --h_feats 128 --predictor_layers 2 --head_layers 2 \
#  --training_subset target --loss focal --epochs 250 --grace 100 --dropout 0.50 --comb_size 1 --flag_rid --batchnorm \
#  --text_emb data/pretrained/"$dset_name"_"$p"_embdi.emb
#done