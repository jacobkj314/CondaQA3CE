export TASK_NAME=negqa

MODEL_NAME=$1 #unifiedqa-v2-t5-11b-1251000
TRAIN_FILES=(unifiedqa)

DATA_DIR="../../data/unifiedqa_formatted_data/"

SEEDS=(70) # 69 68 67 66)

mkdir -p $2
export TEST_FILE=unifiedqa

for SEED in "${SEEDS[@]}"; do
    for SETTING in "${TRAIN_FILES[@]}"; do

      OUTPUT_DIR=$2/${MODEL_NAME}_negation_all_${SEED}_train_${SETTING}_test_${TEST_FILE}
      mkdir -p $OUTPUT_DIR

      deepspeed --num_gpus=1 run_negatedqa_t5.py \
        --model_name_or_path allenai/$MODEL_NAME \
        --train_file ${DATA_DIR}/condaqa_train_unifiedqa.json \
        --validation_file ${DATA_DIR}/condaqa_dev_unifiedqa.json \
        --test_file ${DATA_DIR}/condaqa_test_unifiedqa.json \
        --do_train \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-5 \
        --num_train_epochs 5 \
        --output_dir $OUTPUT_DIR \
        --logging_strategy epoch \
        --evaluation_strategy epoch \
        --report_to wandb \
        --save_strategy epoch \
        --overwrite_cache \
        --seed $SEED \
        --summary_column answer \
        --text_column input \
        --source_prefix "" \
        --max_source_length 512 \
        --max_target_length 100 \
        --overwrite_output_dir \
        --fp16 \
        --deepspeed deepspeed_config.json >${MODEL_NAME}_results_all_${SEED}_train_${SETTING}_test_${TEST_FILE}.txt
    done
done
