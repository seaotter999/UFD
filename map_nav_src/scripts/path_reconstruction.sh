#!/usr/bin/env bash
# Path reconstruction: run the pretrained model on the 500 instructions of each scan,
# build the topology graph incrementally during navigation, then reconstruct paths
# with A* on that graph from the goals given by user feedback.
# Each scan produces user_feedback_data.json and astar_reconstructed_paths.json;
# merging astar_reconstructed_paths.json of all scans and keeping entries whose path
# length is 5~7 gives merged_train_dataset.json used for training.
# Run from map_nav_src: bash scripts/path_reconstruction.sh
# No set -e, so that a single failure does not stop the script
set -uo pipefail

DATA_DIR=../data
SPLIT_DIR="${DATA_DIR}/split_basic"             # split_test_500_<scan>.json (Basic)
OUTPUT_ROOT="../outputs/rec_basic"
# SPLIT_DIR="${DATA_DIR}/split_mixed"     # mixed
# OUTPUT_ROOT="../outputs/rec_mixed"
RESUME_FILE="${DATA_DIR}/ckpts/best_val_unseen"
IMG_FT_FILE="${DATA_DIR}/features/clip_vit-b16_mp3d_hm3d_gibson.hdf5"
ALL_SCANVP_CANDS="${DATA_DIR}/scanvp_cands_relangles_with_habitat.json"
PANO_INPUTS_PATH="${DATA_DIR}/pano_inputs_habitats_more_scan.h5"
CONNECTIVITY_DIR="${DATA_DIR}/connectivity"
CUDA_DEVICES=("0")

# ===== Batch processing =====
shopt -s nullglob
files=( "$SPLIT_DIR"/split_test_500_*.json )

if ((${#files[@]}==0)); then
  echo "No files found: $SPLIT_DIR/split_test_500_*.json"
  exit 1
fi

echo "========================================="
echo "Found ${#files[@]} 500-instruction split files"
echo "========================================="

success_count=0
fail_count=0
skip_count=0
total=${#files[@]}

for i in "${!files[@]}"; do
  new_anno="${files[$i]}"
  base="$(basename "$new_anno")"
  stem="${base%.json}"
  scan_id="${stem#split_test_500_}"
  outdir="$OUTPUT_ROOT/$scan_id"

  echo ""
  echo "[$(($i+1))/$total] Processing: $scan_id"
  echo "File: $new_anno"

  # Skip scans that have already been processed
  if [[ -f "$outdir/user_feedback_data.json" ]] && [[ -f "$outdir/astar_reconstructed_paths.json" ]]; then
    echo "⏭️  Already exists, skipping"
    ((skip_count++))
    continue
  fi

  mkdir -p "$outdir"
  gpu="${CUDA_DEVICES[$(( i % ${#CUDA_DEVICES[@]} )) ]}"
  echo "GPU: $gpu, output: $outdir"

  # Run the Python program
  echo "Starting..."

  # Run each Python call in a subshell to fully isolate it
  (
    CUDA_VISIBLE_DEVICES="$gpu" python -m tool.path_reconstruction \
      --resume_file "$RESUME_FILE" \
      --connectivity_dir "$CONNECTIVITY_DIR" \
      --all_scanvp_cands "$ALL_SCANVP_CANDS" \
      --img_ft_file "$IMG_FT_FILE" \
      --pano_inputs_path "$PANO_INPUTS_PATH" \
      --features clip.b16 \
      --image_feat_size 512 \
      --angle_feat_size 4 \
      --new_anno_dir "$new_anno" \
      --tokenizer bert \
      --enc_full_graph \
      --graph_sprels \
      --fusion dynamic \
      --max_instr_len 200 \
      --max_action_len 15 \
      --output_dir "$outdir" \
      --num_episodes 500
  ) || true  # ignore any error

  # Check the output files
  if [[ -f "$outdir/user_feedback_data.json" ]] && [[ -f "$outdir/astar_reconstructed_paths.json" ]]; then
    echo "✅ Succeeded: $scan_id"
    ((success_count++))
  else
    echo "❌ Failed: $scan_id"
    ((fail_count++))
  fi

  echo "-----------------------------------------"
done

echo ""
echo "========================================="
echo "📊 Summary"
echo "========================================="
echo "  Total: $total files"
echo "  ✅ Succeeded: $success_count"
echo "  ❌ Failed: $fail_count"
echo "  ⏭️  Skipped: $skip_count"
echo "========================================="
echo "Output directory: $OUTPUT_ROOT"
