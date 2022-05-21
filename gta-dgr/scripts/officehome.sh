#!/bin/bash

cd ..

DATA=~/geom-tex-dg/Dassl/data
TRAINER=Vanilla2

D1=art
D2=clipart
D3=product
D4=real_world

for SEED in $(seq 1 5)
do
    for SETUP in $(seq 1 4)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains ${S1} ${S2} ${S3} \
        --target-domains ${T} \
        --dataset-config-file configs/datasets/office_home.yaml \
        --config-file configs/trainers/office_configs/office_${T}.yaml \
        --output-dir output/officehome/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME resnet18_office_${T}
    done
done