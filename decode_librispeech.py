import argparse
import os

import torch
import torchaudio
from tqdm import tqdm

import numpy as np
import pandas as pd
import jiwer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from whisper import whisper
from whisper.whisper.normalizers import EnglishTextNormalizer


class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, download_root, split="test-clean", device="cuda"):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=download_root,
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return (mel, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Running Whisper experiments')
    parser.add_argument('--use_gpt2', action="store_true")
    parser.add_argument('--gpt_kind', type=str, default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument('--lm_weight', type=float, default=0.01)
    parser.add_argument('--ilm_weight', type=float, default=0.005)
    parser.add_argument('--shallow_fusion', action="store_true")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--beam_size', type=int, default=50)
    parser.add_argument('--cache_root', type=str, default="/dataset/.cache")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()

    dataset = LibriSpeech("test-clean", device=device, download_root=args.cache_root)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    model = whisper.load_model("base.en", download_root=args.cache_root).to(device)
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

    for mels, texts in tqdm(loader):
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)

    normalizer = EnglishTextNormalizer()

    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]
    wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    print(f"WER: {wer * 100:.2f} %")
    # GPT 없이 돌리는 것 1
    # GPT랑 돌리는 것 1
    # GPT 없이 돌려서(dev) in-context learning(test)
