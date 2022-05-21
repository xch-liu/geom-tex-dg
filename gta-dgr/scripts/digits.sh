#!/bin/bash

cd ..

DATA=~/geom-tex-dg/Dassl/data
TRAINER=Vanilla2

D1=mnist
D2=syn
D3=svhn
D4=mnist_m

for SEED in $(seq 1 3)
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
        --dataset-config-file configs/datasets/digits_dg.yaml \
        --config-file configs/trainers/digits_configs/digits_dg.yaml \
        --output-dir output/digits/${T}/seed${SEED} \
        --deform-ratio 0.5 \
        MODEL.BACKBONE.NAME cnn_digitsdg_${T}
    done
done