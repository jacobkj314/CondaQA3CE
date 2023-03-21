export TASK_NAME=negqa

MODEL_NAME=$1 #unifiedqa-v2-t5-base-1251000, unifiedqa-v2-t5-large-1251000, unifiedqa-v2-t5-3b-1251000
TRAIN_FILES=(unifiedqa)

DATA_DIR="../../data/unifiedqa_formatted_data/"

SEEDS=(70) #(70 69 68 67 66)

export TEST_FILE=unifiedqa

# Evaluate last model

for SEED in "${SEEDS[@]}"; do
  for SETTING in "${TRAIN_FILES[@]}"; do

      OUTPUT_DIR=$2/${MODEL_NAME}_negation_all_${SEED}_train_${SETTING}_test_${TEST_FILE}      
      mkdir -p $OUTPUT_DIR #Remove? 
      # # # changed test to dev on line 24
      python run_negatedqa_t5.py \
        --model_name_or_path $OUTPUT_DIR \
        --train_file ${DATA_DIR}/condaqa_train_unifiedqa.json \
        --validation_file ${DATA_DIR}/condaqa_dev_unifiedqa.json \
        --test_file ${DATA_DIR}/condaqa_dev_unifiedqa.json \
        --do_eval \
        --do_predict \
        --predict_with_generate \
        --per_device_train_batch_size 8 \
        --learning_rate 1e-5 \
        --num_train_epochs 5 \
        --output_dir $OUTPUT_DIR/test_predictions \
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
        --overwrite_output_dir
      done
done


# Evaluate all checkpoints on dev
for SEED in "${SEEDS[@]}"; do
  for SETTING in "${TRAIN_FILES[@]}"; do

    FILE_DIR=$2/${MODEL_NAME}_negation_all_${SEED}_train_${SETTING}_test_${TEST_FILE}
    for CHECKPOINT in ${FILE_DIR}/checkpoint*; do

      echo ${CHECKPOINT}
      OUTPUT_DIR=${CHECKPOINT}
      mkdir -p $OUTPUT_DIR/val_predictions

      python run_negatedqa_t5.py \
        --model_name_or_path $OUTPUT_DIR \
        --train_file ${DATA_DIR}/condaqa_train_unifiedqa.json \
        --validation_file ${DATA_DIR}/condaqa_dev_unifiedqa.json \
        --test_file ${DATA_DIR}/condaqa_dev_unifiedqa.json \
        --do_eval \
        --do_predict \
        --predict_with_generate \
        --per_device_train_batch_size 8 \
        --learning_rate 1e-5 \
        --num_train_epochs 10 \
        --output_dir $OUTPUT_DIR/val_predictions \
        --logging_strategy epoch\
        --evaluation_strategy epoch\
        --report_to wandb\
        --save_strategy epoch\
        --overwrite_cache\
        --seed $SEED\
        --summary_column answer \
        --text_column input \
        --source_prefix ""\
        --max_source_length 1024\
        --max_target_length 100\
        --overwrite_output_dir > $OUTPUT_DIR/val_predictions/${MODEL_NAME}_results_all_${SEED}_train_${SETTING}_test_${TEST_FILE}_${checkpoint}.txt
      done
done
done
