

#!/bin/bash

# =============================================================================
# Two-stage memory bank cache pipeline
# Stage 1: generate the cache with 500 instructions
# Stage 2: final test on 100 instructions using the cache
# =============================================================================

# Paths
DATA_ROOT="../datasets"
DATA_DIR="../data"    # root directory of data and checkpoints
MODEL_A_PATH="${DATA_DIR}/ckpts/best_val_unseen"
MODEL_B_PATH="${DATA_DIR}/ckpts/best_overall_iter_23500"
CACHE_DIR="../datasets/Cache/multi_scans_cache"
LOG_DIR="../datasets/Cache/multi_scans_logs/cache10"


# Data files
CACHE_GENERATION_FILE="${DATA_DIR}/rec_basic/merged_500_test_dataset.json"  # 500 instructions for generating the cache
FINAL_TEST_FILE="${DATA_DIR}/GSA_Dataset/Test/Residential/Reconstruction/merged_test_dataset.json"      # 100 instructions for the final test

# Cache file
CACHE_FILE="$CACHE_DIR/memory_bank_basic.pkl.gz"

# Create required directories
mkdir -p $CACHE_DIR
mkdir -p $LOG_DIR

echo "=== Stage 1: generate multi-scene cache (500 instructions) ==="

python r2r/main_nav_cache_all_scan.py \
    --root_dir $DATA_ROOT \
    --dataset r2r \
    --output_dir $LOG_DIR/stage1_cache_generation \
    --seed 0 \
    --tokenizer bert \
    --enc_full_graph \
    --graph_sprels \
    --fusion dynamic \
    --num_l_layers 9 \
    --num_x_layers 4 \
    --num_pano_layers 2 \
    --max_traj_num 50 \
    --batch_size 1 \
    --features clip.b16 \
    --image_feat_size 512 \
    --angle_feat_size 4 \
    --anno_dir "${DATA_DIR}/GSA_Dataset" \
    --connectivity_dir "${DATA_DIR}/connectivity" \
    --img_ft_file "${DATA_DIR}/features/clip_vit-b16_mp3d_hm3d_gibson.hdf5" \
    --all_scanvp_cands="${DATA_DIR}/scanvp_cands_relangles_with_habitat.json" \
    --pano_inputs_path="${DATA_DIR}/pano_inputs_habitats_more_scan.h5" \
    --enable_cache \
    --resume_file $MODEL_A_PATH \
    --mode generate_cache \
    --cache_generation_file $CACHE_GENERATION_FILE \
    --cache_file $CACHE_FILE

echo "Stage 1 done: cache saved to $CACHE_FILE"

# echo ""
# echo "=== Stage 2: test with cache (100 instructions) ==="

# python r2r/main_nav_cache_all_scan.py \
#     --root_dir $DATA_ROOT \
#     --dataset r2r \
#     --output_dir $LOG_DIR/stage2_cache_testing \
#     --seed 0 \
#     --tokenizer bert \
#     --enc_full_graph \
#     --graph_sprels \
#     --fusion dynamic \
#     --num_l_layers 9 \
#     --num_x_layers 4 \
#     --num_pano_layers 2 \
#     --max_traj_num 50 \
#     --batch_size 1 \
#     --features clip.b16 \
#     --image_feat_size 512 \
#     --angle_feat_size 4 \
#     --anno_dir "${DATA_DIR}/GSA_Dataset" \
#     --connectivity_dir "${DATA_DIR}/connectivity" \
#     --img_ft_file "${DATA_DIR}/features/clip_vit-b16_mp3d_hm3d_gibson.hdf5" \
#     --all_scanvp_cands="${DATA_DIR}/scanvp_cands_relangles_with_habitat.json" \
#     --pano_inputs_path="${DATA_DIR}/pano_inputs_habitats_more_scan.h5" \
#     --resume_file $MODEL_B_PATH \
#     --mode test_with_cache \
#     --cache_testing_file $FINAL_TEST_FILE \
#     --multi_scene_cache_file $CACHE_FILE \
#     --detailed_output

# echo ""
# echo "=== Both stages finished! ==="
# echo "Outputs:"
# echo "  - Cache file: $CACHE_FILE"
# echo "  - Stage 1 logs: $LOG_DIR/stage1_cache_generation/"
# echo "  - Stage 2 logs and results: $LOG_DIR/stage2_cache_testing/"
