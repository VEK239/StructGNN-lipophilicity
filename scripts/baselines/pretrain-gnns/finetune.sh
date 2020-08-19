#!/bin/bash

i=./model_gin

for k in contextpred.pth edgepred.pth infomax.pth masking.pth supervised.pth supervised_contextpred.pth supervised_edgepred.pth supervised_infomax.pth supervised_masking.pth
do
	~/anaconda3/envs/chemprop/bin/python finetune.py --filename $k --input_model_file $i --gnn_type gin --epochs 2000
	echo $k
done

i=./model_architecture

for k in gat_contextpred.pth gat_supervised.pth gat_supervised_contextpred.pth
do
        ~/anaconda3/envs/chemprop/bin/python finetune.py --filename $k --input_model_file $i --gnn_type gat --epochs 2000
        echo $k
done

for k in  gcn_supervised.pth gcn_supervised_contextpred.pth gcn_contextpred.pth
do
	~/anaconda3/envs/chemprop/bin/python finetune.py --filename $k --input_model_file $i --gnn_type gcn --epochs 2000
        echo $k
done

for k in  graphsage_supervised.pth graphsage_supervised_contextpred.pth graphsage_contextpred.pth
do
        ~/anaconda3/envs/chemprop/bin/python finetune.py --filename $k --input_model_file $i --gnn_type graphsage --epochs 2000
        echo $k
done

