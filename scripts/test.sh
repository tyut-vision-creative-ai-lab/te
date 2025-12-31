# bash ./scripts/test.sh

# test our large model
python test.py \
    --model_name HLFRN \
    --sigma 50 \
    --modelPath ./pretrained_models/HLFRN/model_sigma_50.pth \
    --dataPath  ./datasets/test_noiseLeve_10-20-50_4-11_5x5.mat \
    --savePath ./results/sythesis_img_test/ \


# test our small model
python test.py \
    --model_name HLFRN \
    --sigma 50 \
    --modelPath ./pretrained_models/HLFRN/model_sigma_50_S.pth \
    --dataPath  ./datasets/test_noiseLeve_10-20-50_4-11_5x5.mat \
    --savePath ./results/sythesis_img_test/ \
    --n_groups 3 \
    --n_blocks 3 


# python test.py \
#     --model_name DRLF \
#     --sigma 50 \
#     --modelPath ./pretrained_models/DRLF/model_sigma_50.pth \
#     --dataPath  ./datasets/test_noiseLeve_10-20-50_4-11_5x5.mat \
#     --savePath ./results/sythesis_img_test/ \

# python test.py \
#     --model_name MSP \
#     --sigma 50 \
#     --modelPath ./pretrained_models/MSP/model_sigma_50.pth \
#     --dataPath  ./datasets/test_noiseLeve_10-20-50_4-11_5x5.mat \
#     --savePath ./results/sythesis_img_test/MSP_20_v1 \


# python test.py \
#     --model_name PFE \
#     --sigma 50 \
#     --modelPath ./pretrained_models/PFE/model_sigma_50.pth \
#     --dataPath  ./datasets/test_noiseLeve_10-20-50_4-11_5x5.mat \
#     --savePath ./results/sythesis_img_test/ \