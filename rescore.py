import argparse
import logging
import os
import random
import sys
from io import StringIO
from typing import List

import torch
from tqdm import tqdm

from data import DataIter, Dictionary
from knowledge.cilin import CiLin

logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(funcName)s\t%(message)s", datefmt="%H:%M:%S")


def extract_sents(file_name: str) -> List[str]:
    _sents = []
    with open(file_name, encoding="utf8") as f:
        for line in f:  # 类似：15  -925.24	14.509	 <s> 那 还有 什么 </s>
            _token, _ac_score, _lm_score, *sent = line.split()
            if sent[0] == "<s>":
                sent = sent[1:]
            if sent[-1] == "</s>":
                sent = sent[:-1]
            sent = [x for x in sent if x != "<SPOKEN_NOISE>"]
            _sents.append(" ".join(sent))
    return _sents


def replace_OOV(sentences: List[str], vocab: Dictionary, cilin: CiLin, level: int = 5) -> List[str]:
    """ 将句子中的OOV替换为同义词 """
    _sents = []
    for sent in sentences:
        words = sent.split()
        for j, w in enumerate(words):
            if w in vocab:
                continue
            # OOV使用词林猜词
            _, similar_words = cilin.guess_group(w, level=level)
            similar_words = list(filter(lambda w: w in vocab, similar_words))
            if len(similar_words) > 0:
                words[j] = sorted(similar_words, key=lambda x: vocab.word2idx[x])[-1]
                # words[j] = random.choice(similar_words)
        _sents.append(" ".join(words))
    return _sents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True, help="模型位置")
    parser.add_argument("--file", "-i", type=str, required=True, help="保存需要重打分的文件路径")
    parser.add_argument("--out_file", "-o", type=str, default=None, help="打分结果输出位置")
    parser.add_argument("--dict", "-d", type=str, required=True, help="字典")
    parser.add_argument("--cilin", action="store_true", help="是否使用词林")
    parser.add_argument("--cuda", "-c", type=int, default=0, help="使用什么设备")
    args = parser.parse_args()

    if args.cilin:
        logging.info("Load CiLin...")
        cilin = CiLin()

    logging.info("Load model...")
    CUDA = False
    if args.cuda >= 0:
        torch.cuda.set_device(args.cuda)
        CUDA = True
    device = "cpu" if (args.cuda == -1) else f"cuda:{args.cuda}"
    model = torch.load(args.model, map_location=torch.device(device)).eval()

    logging.info("Build dictionary...")
    vocab = Dictionary(args.dict)

    logging.info("Evaluating...")
    nbest_sents = extract_sents(args.file)
    if args.cilin:
        nbest_sents = replace_OOV(nbest_sents, vocab, cilin)  # 词林猜词
    f_out = open(args.out_file, "w", encoding="utf8") if args.out_file else sys.stdout
    
    data_iter = DataIter(StringIO("\n".join(nbest_sents)), 1, vocab, CUDA, False)
    for inputs, target, length in tqdm(data_iter):
        loss = model.loss((inputs, target, length)).detach().cpu().item()
        score = loss * length[0] * 2.3
        f_out.write(f"{str(score)}\n")
