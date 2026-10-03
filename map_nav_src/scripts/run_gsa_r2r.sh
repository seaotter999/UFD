# Run from map_nav_src: bash scripts/run_gsa_r2r.sh
DATA_DIR=../data    # root directory of data and checkpoints

train_alg=dagger

features=clip.b16
ft_dim=512

ngpus=1
seed=0

WANDB_PROJECT="GR-DUET"
WANDB_RUN_NAME="ufd_basic"
name=${WANDB_RUN_NAME}

outdir=../datasets/R2R/exprs_map/finetune/${name}
anno_dir=${DATA_DIR}/GSA_Dataset
aug=${anno_dir}/Train/prevalent_aug_train_enc.json
new_anno_dir=${DATA_DIR}/rec_basic                # merged_train_dataset.json: training set of A*-reconstructed paths (Basic)
# new_anno_dir=${DATA_DIR}/rec_mixed        # mixed
connectivity_dir=${DATA_DIR}/connectivity
img_ft_file=${DATA_DIR}/features/clip_vit-b16_mp3d_hm3d_gibson.hdf5
all_scanvp_cands=${DATA_DIR}/scanvp_cands_relangles_with_habitat.json
pano_inputs_path=${DATA_DIR}/pano_inputs_habitats_more_scan.h5
resume_file=${DATA_DIR}/ckpts/best_val_unseen          # official GR-DUET checkpoint released with GSA-VLN

flag="--root_dir ${DATA_DIR}
    --dataset r2r
    --output_dir ${outdir}
    --world_size ${ngpus}
    --seed ${seed}
    --tokenizer bert

    --enc_full_graph
    --graph_sprels
    --fusion dynamic

    --expert_policy spl
    --train_alg ${train_alg}

    --num_l_layers 9
    --num_x_layers 4
    --num_pano_layers 2

    --max_action_len 15
    --max_instr_len 200

    --batch_size 2
    --lr 1e-5
    --iters 100000
    --log_every 1000
    --optim adamW

    --features ${features}
    --image_feat_size ${ft_dim}
    --angle_feat_size 4

    --ml_weight 0.2

    --feat_dropout 0.4
    --dropout 0.5
    --all_scanvp_cands ${all_scanvp_cands}
    --pano_inputs_path ${pano_inputs_path}

    --new_anno_dir ${new_anno_dir}
    --anno_dir ${anno_dir}
    --connectivity_dir ${connectivity_dir}
    --img_ft_file ${img_ft_file}

    --use_new_dataset
    --wandb_project ${WANDB_PROJECT}
    --wandb_run_name ${WANDB_RUN_NAME}
    --gamma 0.
    --aug  ${aug}
    "

# train
python r2r/main_nav_with_all_scan.py $flag \
    --resume_file ${resume_file} \
    --max_traj_num 50

#
# eval
# MODEL_DIR="../datasets/R2R/exprs_map/finetune/dagger-clip.b16-seed.0-init.aug.45k/ckpts"
# for model in "$MODEL_DIR"/*; do
#     echo "Evaluating model: $model"
#     CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag \
#         --tokenizer bert \
#         --resume_file "$model" \
#         --test
# done

