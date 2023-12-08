python3 decode_librispeech.py \
    --batch_size 1 \
    --use_gpt \
    --gpt_kind gpt2 \
    --shallow_fusion \
    --use_icl \
    --index_path ./bert_index.faiss \
    --csv_path ./base_dev_wrong.csv \
    --dataset_offset 1400 \
    --cache_root /Users/dhdroid/.cache
