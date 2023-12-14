from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch


from collections import defaultdict, Counter
from string import ascii_lowercase
import re
import numpy as np

def prefix_beam_search(ctc, alphabet, lm=None, k=27, alpha=0.30, beta=5, prune=0.001):
    lm = (lambda l: 1) if lm is None else lm # if no LM is provided, just set to function returning 1
    W = lambda l: re.findall(r'\w+[\s|>]', l)
    F = ctc.shape[1]
    ctc = np.vstack((np.zeros(F), ctc)) # just add an imaginative zero'th step (will make indexing more intuitive)
    T = ctc.shape[0]
	
    O = ''
    Pb, Pnb = defaultdict(Counter),defaultdict(Counter)
    Pb[0][O] = 1
    Pnb[0][O] = 0
    A_prev = [O]
	# END: STEP 1

	# STEP 2: Iterations and pruning
    for t in range(1, T):
        pruned_alphabet = [alphabet[i] for i in np.where(ctc[t] > prune)[0]]
        for l in A_prev:
			
            if len(l) > 0 and l[-1] == '>':
                Pb[t][l] = Pb[t - 1][l]
                Pnb[t][l] = Pnb[t - 1][l]
                continue  

            for c in pruned_alphabet:
                c_ix = alphabet.index(c)
				# END: STEP 2
				
				# STEP 3: “Extending” with a blank
                if c == '%':
                    Pb[t][l] += ctc[t][-1] * (Pb[t - 1][l] + Pnb[t - 1][l])
				# END: STEP 3
				
				# STEP 4: Extending with the end character
                else:
                    l_plus = l + c
                    if len(l) > 0 and c == l[-1]:
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
                        Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
				# END: STEP 4

					# STEP 5: Extending with any other non-blank character and LM constraints
                    elif len(l.replace(' ', '')) > 0 and c in (' ', '>'):
                        lm_prob = lm(l_plus.strip(' >')) ** alpha
                        Pnb[t][l_plus] += lm_prob * ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    else:
                        Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
					# END: STEP 5

					# STEP 6: Make use of discarded prefixes
                    if l_plus not in A_prev:
                        Pb[t][l_plus] += ctc[t][-1] * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pnb[t - 1][l_plus]
					# END: STEP 6

		# STEP 7: Select most probable prefixes
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]
    print("PREV:", A_prev[0], A_prev[1], A_prev[2])
    return A_prev[0].strip('>')

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-100h")

alphabet = [processor.decode([i]).lower() for i in range(32)]
alphabet[0] = "@"
alphabet[4] = " "

for i in range(1):
    audio_sample = dataset[i]
    print("ans:", audio_sample["text"].lower())
    inputs = processor(audio_sample["audio"]["array"], sampling_rate=audio_sample["audio"]["sampling_rate"], return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        logits = logits[0]
    probs = torch.softmax(logits, dim=-1)

    print("prediction:", prefix_beam_search(probs, alphabet).replace("@", ""))
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = processor.batch_decode(predicted_ids)

# transcription[0].lower()

