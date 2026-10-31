# ClassNNLM

<p>
  <a href="https://storage.googleapis.com/lacuna-pdfs-lunar-airport-385023/pdfs/c4/d7/c4d712aa19908dfd403ef48482a348c262571862fe567583dc112b99afbf5726.pdf"><img src="https://img.shields.io/badge/Paper-Interspeech%202021-blue" alt="Paper"></a>
  <a href="https://www.isca-archive.org/interspeech_2021/dai21b_interspeech.html"><img src="https://img.shields.io/badge/ISCA-Archive-lightgrey" alt="ISCA"></a>
  <a href="https://www.superlectures.com/interspeech2021/class-based-neural-network-language-model-for-second-pass-rescoring-in-asr-3-minutes-introduction?id=279&lang=en"><img src="https://img.shields.io/badge/Talk-Interspeech%202021-red" alt="Talk"></a>
  <img src="https://img.shields.io/badge/Python-3.7%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.8%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
</p>

Official implementation of **"Class-Based Neural Network Language Model for Second-Pass
Rescoring in ASR"** (Interspeech 2021, pp. 2022–2026).

In second-pass ASR rescoring, the NNLM is trained separately from the ASR system, so their
vocabularies mismatch: the n-best hypotheses contain words the rescoring LM has never seen.
The standard fix — mapping every OOV word to a single `UNK` token — is both ambiguous and
statistically broken (all OOV mass piles onto one token). Simply enlarging the vocabulary
doesn't work either, because word frequencies are long-tailed. This repo implements the
paper's alternative: **give every OOV word a home in a well-trained word class, and let it
inherit the class's probability.**

## Key ideas

1. **Class-based softmax.** Factorize the output as
   `P(w|h) = P(c_w|h) · P(w|c_w, h)` — first predict the word's class, then the word
   within the class (`decoder.py: ClassBasedSMDecoder`).
2. **Two ways to build the classes.**
   - *Knowledge-driven*: Chinese Synonym Cilin （同义词词林） tree (depth 5); classes are
     nodes at a chosen depth, similarity via information content and lowest common ancestor
     (`knowledge/cilin.py`).
   - *Data-driven*: word2vec embeddings + k-means (`train_w2v.py`).
3. **OOV words inherit from their class.** An OOV word is assigned to its most similar
   class (words missing from Cilin are placed by decomposing them into characters/sub-words
   and matching those — the "simple guessing" method, with optional OpenHowNet support).
   Its parameters are estimated as a weighted average of the class members'
   (`build_vocab_with_class.py`), weighted by `1/(count(w)+1)` so the estimate resembles
   the class's *low-frequency* words rather than its dominant ones.

Experiments in the paper show this significantly reduces the performance degradation
caused by vocabulary mismatch during NNLM rescoring.

## Repository layout

| Path | What it is |
|---|---|
| `model.py` | LSTM NNLM (`RNNModel`) |
| `decoder.py` | `SMDecoder` and `ClassBasedSMDecoder` (class-based softmax) |
| `knowledge/cilin.py` | Cilin-based clustering, IC/LCA similarity, OOV class assignment |
| `build_vocab_with_class.py` | Vocabulary construction with classes (`class_assign_by_pos`, `class_assign_by_tree`) |
| `train.py` / `train_w2v.py` | NNLM training / word2vec training for data-driven clustering |
| `rescore.py` | Second-pass n-best rescoring |
| `data.py` / `utils.py` / `alias_multinomial.py` | Data pipeline and helpers |

Related: [MultiClassLM](https://github.com/Randool/MultiClassLM) — the class-based
*n-gram* counterpart with absolute discounting.

## Rescoring

```bash
python rescore.py \
  --model ./params/model_4w_sm.pt \
  --vocab ./data/sms/vocab4w.c.txt \
  --nbest <nbest-file>
```

See `README_new.md` (Chinese) for the original development notes: file-by-file changes,
the OOV guessing method, and the Cilin coverage stats (75.5% on the sms vocabulary).

## Citation

```bibtex
@inproceedings{dai21_interspeech,
  title={Class-Based Neural Network Language Model for Second-Pass Rescoring in ASR},
  author={Dai, Lingfeng and Liu, Qi and Yu, Kai},
  booktitle={Proc.\ Interspeech 2021},
  pages={2022--2026},
  year={2021},
  address={Brno, Czechia},
  doi={10.21437/Interspeech.2021-1080}
}
```

## Authors

Lingfeng Dai$^{\#}$, Qi Liu$^{\#}$, Kai Yu$^{*}$ ($^{\#}$equal contribution)
— X-LANCE Lab, Shanghai Jiao Tong University
