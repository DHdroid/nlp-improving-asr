A project for 2023-Fall SNU NLP lecture (Team 5)

This codebase is built on [Whisper](https://github.com/openai/whisper) and [WhisperBiasing](https://github.com/BriansIDP/WhisperBiasing)

# Method Overview

## Shallow Fusion
Utillize the information from the LM outputs as follows: (AM for Audio Model, LM for Language Model)

$$\text{score} = log P_{\text{AM}}(Y|X) + \lambda{}\cdot{}logP_{\text{LM}}(Y)$$

In this project, we used `Whisper-base.en` for AM, and `GPT-2-small` for LM.

## Few-shot Prompted Shallow Fusion (Proposed Method 1)
Give LM few-shot examples to 1) provide LM with the following context and 2) leverage in-context learning ability of LM.

$$\text{score} = log P_{\text{AM}}(Y|X) + \lambda{}\cdot{}logP_{\text{LM}}(Y | \text{few-shot prompt})$$

## Combined Shallow Fusion (Proposed Method 2)
- Generate first $K$ tokens with Few-shot Prompted Shallow Fusion
- Generate the remaining tokens with naive shallow fusion

$$
    \text{score} = 
        \begin{cases}
            log P_{\text{AM}}(Y|X) + \lambda{}\cdot{}logP_{\text{LM}}(Y | \text{few-shot prompt}), & \text{if $\text{len}(Y) \leq{} K$,}  \\
            log P_{\text{AM}}(Y|X) + \lambda{}\cdot{}logP_{\text{LM}}(Y), & \text{otherwise,}
        \end{cases}
$$


## Experimental Results on LibriSpeech
Used beam_size=5, $\lambda{}$=0.05 in all experiments. Random retrieval is used for few-shot prompting.
|Methods|test-clean (WER)|test-other (WER)|Average|
|---|---|---|---|
|Whisper|4.35|9.42|6.89|
|Shallow Fusion|**3.97**|9.46|6.72|
|Few-shot Prompted Shallow Fusion|4.26|9.42|6.84|
|Combined Shallow Fusion|4.0|**9.23**|**6.62**|


# How to run our codes

## Install Dependencies
```
pip install -r requirements.txt
```
## Example
### 1. Generate data pool
Example command :
```
python3 decode_librispeech.py \
    --batch_size 1 \
    --beam_size 5 \
    --split $YOUR_SPLIT \
    --output_path $YOUR_OUTPUT_PATH \
    --cache_root $YOUR_CACHE_ROOT \
```
### 2. Filter the data pool(Optional, Recommended)
Use `filter_csv.sh`.
In this file, replace input csv path, output csv path to yours and select filtering parameter min_wer, max_wer.

### 3. Generate vector DB(Optional)
If you want to retrieve examples using similarity search, you have to build your faiss index by calling `search_sentence.py`.

### 4. Inference
Example command :
```
python3 decode_librispeech.py \
    --batch_size 1 \
    --beam_size 5 \
    --use_gpt \
    --gpt_kind gpt2 \
    --shallow_fusion \
    --use_icl \
    --sample_random \
    --split $YOUR_SPLIT \
    --output_path $YOUR_OUTPUT_PATH \
    --cache_root $YOUR_CACHE_ROOT \
```


### Arguments

`--whisper_model` : Kind of whisper model. Default is "base.en"  
`--split` : The dataset type. Default is "test-clean"  
`--use_gpt2` : Select whether to use lm or not. Default is "False"  
`--gpt_kind` : The model of lm. Default is "gpt2"  
`--lm_weight` : The weight of shallow fusion. Default is "0.05"  
`--ilm_weight` : The weight of internal lm weight. In our project this argument should be 0. Default is "0"  
`--shallow_fusion` : Select whether to use shallow fusion. Default is "False"  
`--batch_size` : Batch size. Default is "1"  
`--beam_size` : Depth of beam search.. Default is "5"  
`--cache_root` : Cache_root.  
`--num_data` : Number of datas to decode. When this argument is "-1", decode all datas. Default is "-1"  
`--dataset_offset` : Dataset offset. Default is "0"  
`--use_icl` : Select whether to use few-shot prompting. Default is "False"  
`--index_path` : Path of index file.  
`--csv_path` : Path of csv file.  
`--output_path` : Path of output file.  
`--num_examples` : Number of prompt examples. Default is "10"  
`--sample_random` : Select whether to create prompt randomly or not. Default is "False"  
`--prefix_lenght` : Lenght of prefix tokens. Default is "3"  

