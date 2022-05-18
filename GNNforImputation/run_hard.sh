#!/bin/bash

#dset_name=$1

# XE Loss

# Linear shared, attention heads
for dset_name in 'beer' 'bikes' 'bikes-dekho' 'bikes-wale'
do
clean_dataset=data/new-hard/new-hard/$dset_name.csv
for p in 2 10 20
do
  f=data/new-hard/new-hard_dirty/$dset_name'_all_columns_'$p'.csv'
  for emb in "data/pretrained/""$dset_name""_all_columns_""$p""_embdi_f4.emb" "data/pretrained/""$dset_name""_all_columns_""$p""_ft.emb"
  do
    for ncol in  {1..2}
    do
    echo "###########################" $f
    python main_multilabel.py --ground_truth $clean_dataset \
    --dirty_dataset $f \
    --architecture multitask --graph_layers 2 --gnn_feats 64 --h_feats 128 --predictor_layers 2 --head_layers 2 \
    --training_subset target --loss xe  --epochs 1000 --grace 100 --dropout_clf 0.2 \
    --text_emb $emb --max_components 64 --head_model attention --shared_model linear
    cp /content/GNNforImputation/results/results.csv   /content/drive/MyDrive/Colab\ Notebooks/results.csv
   done
  done
  for emb in "data/pretrained/""$dset_name""_all_columns_""$p""_embdi_f4.emb" "data/pretrained/""$dset_name""_all_columns_""$p""_ft.emb"
  do
    for ncol in  {1..2}
    do
    echo "###########################" $f
    python main_multilabel.py --ground_truth $clean_dataset \
    --dirty_dataset $f \
    --architecture multitask --graph_layers 2 --gnn_feats 64 --h_feats 128 --predictor_layers 2 --head_layers 2 \
    --training_subset target --loss xe  --epochs 1000 --grace 100 --dropout_clf 0.2 \
    --text_emb $emb --max_components 64 --head_model attention --shared_model linear --flag_col
    cp /content/GNNforImputation/results/results.csv   /content/drive/MyDrive/Colab\ Notebooks/results.csv
   done
  done

done
done
