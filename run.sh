#!/bin/bash

cd projects/Neu-Review-Rec
source activate pytorch_env


echo "first begin"
python main_kg.py train --model=DAML --dataset=largecellphone_kg_data --num_fea=2 > log/train_DAML_KG_fm_cellphone_3 &

if [$? -ne 0]; then
    echo "faild"
else
    echo "success second"
    python main_kg.py train --model=DAML --dataset=largeautomotive_kg_data --num_fea=2 > log/train_DAML_KG_fm_automotive &

fi

if [$? -ne 0]; then
    echo "faild"
else
    echo "success therd"
    python main_kg.py train --model=DAML --dataset=largeclothing_kg_data --num_fea=2 > log/train_DAML_KG_fm_clothing &
fi