# nlp-improving-asr
A project for 2023-2 SNU NLP lecture (Team 5)

## In-Context Learning
설명~~~~

## Dependencies
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

