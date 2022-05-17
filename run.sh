#!/bin/bash

python plnlp_sign.py --data_name=ogbl-collab  --predictor=DOT --use_valedges_as_input=True --year=2010 --train_on_subgraph=True --epochs=800 --eval_last_best=True --dropout=0.3 --gnn_num_layers=1 --grad_clip_norm=1 --use_lr_decay=True --random_walk_augment=True --walk_length=10 --loss_func=WeightedHingeAUC --data_path=dataset