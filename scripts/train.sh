# bash ./scripts/train.sh

python train.py \
    --model_name HLFRN \
    --sigma 20 \
    --dataPath  ./datasets/train_noiseLevel_10-20-50_4-11_color_5x5.mat \
    --saveCheckpointsDir ./checkpoints/ \

# python train.py \
#     --model_name DRLF \
#     --sigma 20 \
#     --dataPath  ./datasets/train_noiseLevel_10-20-50_4-11_color_5x5.mat \
#     --saveCheckpointsDir ./checkpoints/ \

# python train.py \
#     --model_name MSP \
#     --sigma 20 \
#     --dataPath  ./datasets/train_noiseLevel_10-20-50_4-11_color_5x5.mat \
#     --saveCheckpointsDir ./checkpoints/ \

# python train.py \
#     --model_name PFE \
#     --sigma 20 \
#     --dataPath  ./datasets/train_noiseLevel_10-20-50_4-11_color_5x5.mat \
#     --saveCheckpointsDir ./checkpoints/ \
   