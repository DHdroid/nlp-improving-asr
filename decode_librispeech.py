import argparse
import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import jiwer

from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from whisper import whisper
from whisper.whisper.normalizers import EnglishTextNormalizer
from search_sentence import read_data_from_csv, initialize_or_load_faiss_index, search_similar_sentence, get_bert_tokenizer_model, search_random_sentence
from generate_prompt import generate_gpt2_prompt

class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, download_root, split="test-clean", device="cuda", num_data=-1, dataset_offset=0):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=download_root,
            url=split,
            download=True,
        )
        self.dataset_offset = dataset_offset
        self.device = device
        self.num_data = num_data

    def __len__(self):
        if self.num_data == -1 : return len(self.dataset) - self.dataset_offset
        return self.num_data

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item + self.dataset_offset]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        return (mel, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Running Whisper experiments')
    parser.add_argument('--whisper_model', type=str, default="base.en")
    parser.add_argument('--split', type=str, default="test-clean")
    parser.add_argument('--use_gpt2', action="store_true")
    parser.add_argument('--gpt_kind', type=str, default="gpt2")
    parser.add_argument('--lm_weight', type=float, default=0.05)
    parser.add_argument('--ilm_weight', type=float, default=0.0)
    parser.add_argument('--shallow_fusion', action="store_true")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--cache_root', type=str, default="/dataset/.cache")
    parser.add_argument('--num_data', type=int, default=-1)
    parser.add_argument('--dataset_offset', type=int, default=0)
    parser.add_argument('--use_icl', action="store_true")
    parser.add_argument('--index_path', type=str)
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--num_examples', type=int, default=10)
    parser.add_argument('--sample_random', action="store_true")
    parser.add_argument('--prefix_length', type=int, default=3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()
    if args.use_icl:
        args.batch_size = 1
        loaded_hypotheses, loaded_references = read_data_from_csv(args.csv_path)
        bert_tokenizer, bert_model = get_bert_tokenizer_model()
        index = initialize_or_load_faiss_index(loaded_hypotheses, args.index_path)
        if not args.use_gpt2:
            print("You should set use_gpt2 as True when trying to use ICL")
            exit()

    dataset = LibriSpeech(args.cache_root, args.split, device=device, num_data=args.num_data, dataset_offset=args.dataset_offset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    model = whisper.load_model(args.whisper_model, download_root=args.cache_root).to(device)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    if args.use_gpt2:
        generator = pipeline('text-generation', model=args.gpt_kind, device=device)
        gpt_model = generator.model
        gpt_tokenizer = generator.tokenizer
        ilme_model = whisper.load_model("base.en", download_root=args.cache_root).eval().to(device)
    else:
        gpt_model = None
        gpt_tokenizer = None
        ilme_model = None
    
    if args.use_icl:
        option1 = whisper.DecodingOptions(language="en",
                                          without_timestamps=True,
                                          beam_size=args.beam_size,
                                          fp16=(device=="cuda"))
        
        option2 = whisper.DecodingOptions(language="en",
                                          without_timestamps=True,
                                          useGPT=args.use_gpt2,
                                          GPT2=gpt_model,
                                          GPT2tokenizer=gpt_tokenizer,
                                          shallowfusion=True,
                                          ilme_model=ilme_model,
                                          lm_weight=args.lm_weight,
                                          ilm_weight=args.ilm_weight,
                                          beam_size=args.beam_size,
                                          fp16=(device=="cuda"))
    if args.shallow_fusion:
        options = whisper.DecodingOptions(language="en",
                                          without_timestamps=True,
                                          useGPT=args.use_gpt2,
                                          GPT2=gpt_model,
                                          GPT2tokenizer=gpt_tokenizer,
                                          shallowfusion=True,
                                          ilme_model=ilme_model,
                                          lm_weight=args.lm_weight,
                                          ilm_weight=args.ilm_weight,
                                          beam_size=args.beam_size,
                                          fp16=(device=="cuda"))
    else:
        options = whisper.DecodingOptions(language="en",
                                          without_timestamps=True,
                                          beam_size=args.beam_size,
                                          fp16=(device=="cuda"))
    hypotheses = []
    references = []
    audio_urls = []
    normalizer = EnglishTextNormalizer()
    if args.use_icl:
        for mels, texts in tqdm(loader):
            results = model.decode(mels, option1)
            predicted = results[0].text
            # search
            if args.sample_random: 
                retrieved = search_random_sentence(loaded_hypotheses, loaded_references, args.num_examples)
            else : 
                retrieved = search_similar_sentence(index, predicted, loaded_hypotheses, loaded_references, bert_tokenizer, bert_model, args.num_examples)
            # prompt
            prompt = generate_gpt2_prompt(retrieved, predicted, gpt_tokenizer, normalizer, 1024)
            prompted_results = model.decode(mels, option2, prompt, args.prefix_length)

            original = predicted
            new = prompted_results[0].text
            hypotheses.extend([result.text for result in prompted_results])
            references.extend(texts)
    else:
        for mels, texts in tqdm(loader):
            results = model.decode(mels, options)
            hypotheses.extend([result.text for result in results])
            references.extend(texts)


    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
    data.to_csv(path_or_buf=args.output_path)
    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]
    wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    print(f"WER: {wer * 100:.2f} %")
