export CUDA_VISIBLE_DEVICES=0,1

DELTA=delta
ALPHA=alpha
LOW_BOUND=low_bound
TOP_BOUND=top_bound
SEG_SIZE=640
MODEL=qwen_audio_dir
SOURCE=translation_file/source_audio.txt
TARGET=translation_file/target.txt

simuleval --agent contrastive_policy.py \
    --source-segment-size $SEG_SIZE \
    --source_size $SEG_SIZE \
    --source $SOURCE \
    --target $TARGET \
    --threshold $ALPHA \
    --low_bound $LOW_BOUND \
    --top_bound $TOP_BOUND \
    --decision_ratio $DELTA \
    --lang_pair fr_en \
    --model_dir $MODEL \
    --output result_log_${SEG_SIZE}_${LOW_BOUND}_${TOP_BOUND}_${DELTA}_${ALPHA}