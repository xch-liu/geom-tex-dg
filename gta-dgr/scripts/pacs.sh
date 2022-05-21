#!/bin/bash

cd ..

DATA=~/geom-tex-dg/Dassl/data
TRAINER=Vanilla2

D1=art_painting
D2=cartoon
D3=photo
D4=sketch

for SEED in $(seq 4 5)
do
    for SETUP in $(seq 1 4)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D3}
            S2=${D2}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D4}
            S2=${D3}
            S3=${D1}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D4}
            S3=${D2}
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
        --dataset-config-file configs/datasets/pacs.yaml \
        --config-file configs/trainers/pacs_configs/pacs_${T}.yaml \
        --output-dir output/pacs/${T}/seed${SEED} \
        --deform-ratio 0.5 \
        MODEL.BACKBONE.NAME resnet18_pacs_${T}
    done
donenee
done