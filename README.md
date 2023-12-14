A project for 2023-Fall SNU NLP lecture (Team 5)

This codebase is built on [Whisper](https://github.com/openai/whisper) and [WhisperBiasing](https://github.com/BriansIDP/WhisperBiasing)

## Shallow Fusion
Utillize the information from the LM outputs as follows: (AM for Audio Model, LM for Language Model)

$log P_{\text{AM}}(Y|X) + \lambda{}\cdot{}logP_{\text{LM}}(Y)$

In this project, we used `Whisper-base.en` for AM, and `GPT-2-small` for LM.

## Few-shot Prompted Shallow Fusion
Give LM few-shot examples to 1) provide LM with the following context and 2) leverage in-context learning ability of LM.

$log P_{\text{AM}}(Y|X) + \lambda{}\cdot{}logP_{\text{LM}}(Y | \text{few-shot prompt})$

## Combined Shallow Fusion (Our proposed method)
- Generate first $K$ tokens with Few-shot Prompted Shallow Fusion
- Generate the remaining tokens with naive shallow fusion

## Experimental Results
|제목 셀1|제목 셀2|제목 셀3|제목 셀4|
|---|---|---|---|
|Shallow Fusion|내용 2|내용 3|내용 4|
|내용 5|내용 6|내용 7|내용 8|
|내용 9|내용 10|내용 11|내용 12|

# How to run our codes

## Install Dependencies
```
pip install -r requirements.txt
```
## Example
Use `run_icl.sh`.
In file, you should replace the spilt, output_path, cache_root to yours.
```
./run_icl.sh
```

### Arguments

`--whisper_model` : Kind of whisper model. Default is "base.en"  
`--split` : The dataset type. Default is "test-clean"  
`--use_gpt2` : Select whether to use lm or not. Default is "False"  
`--gpt_kind` : The model of lm. Default is "gpt2"  
`--lm_weight` : The weight of shallow fusion. Default is "0.05"  
`--ilm_weight` : The weight of internal lm weight. In our project this argument should be 0. Default is "0"  
`--shallow_fusion` : Select whether to sue shallow fusion. Default is "False"  
`--batch_size` : Batch size. Default is "1"  
`--beam_size` : Depth of beam search.. Default is "5"  
`--cache_root` : Cache_root.  
`--num_data` : Number of datas to decode. When this argument is "-1", decode all datas. Default is "-1"  
`--dataset_offset` : Dataset offset. Default is "0"  
`--use_icl` : Select whether to us in-context learning. Default is "False"  
`--index_path` : Path of index file.  
`--csv_path` : Path of csv file.  
`--output_path` : Path of output file.  
`--num_examples` : Number of prompt examples. Default is "10"  
`--sample_random` : Select whether to create prompt randomly or not. Default is "False"  
`--prefix_lenght` : Lenght of prefix tokens. Default is "3"  

