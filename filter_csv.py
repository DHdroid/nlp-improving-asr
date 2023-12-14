from whisper.whisper.normalizers import EnglishTextNormalizer
import pandas as pd
import jiwer
import json
import argparse


parser = argparse.ArgumentParser(description = 'Running filter experiments')
parser.add_argument('--input_path', type=str, default="./reference.csv")
parser.add_argument('--output_path', type=str, default="./output.csv")
parser.add_argument('--min_wer', type=int, default=0)
parser.add_argument('--max_wer', type=int, default=0.5)


args = parser.parse_args()


normalizer = EnglishTextNormalizer()

original = pd.read_csv(args.input_path)
cnt = 0
error_idx = []
for i, row in original.iterrows():
    wer = jiwer.wer(normalizer(row["hypothesis"]), normalizer(row["reference"]))
    if wer == 0 or wer < args.min_wer or wer > args.max_wer:
        cnt += 1
        error_idx.append(i)

wrong_only = original.drop(error_idx)
wrong_only.reset_index(drop=True)
wrong_only.to_csv(args.output_path, index=False)