#!/usr/bin/env python
# encoding: utf-8
import argparse
import random
from collections import Counter, defaultdict

from tqdm import tqdm

from knowledge.cilin import CiLin, level2len


def freq_counter(corpus, max_vocab_size: int = None, min_count: int = None):
    with open(corpus, encoding="utf8") as f:
        counter = Counter(f.read().split())

    if max_vocab_size or min_count:
        max_vocab_size = max_vocab_size if (max_vocab_size and max_vocab_size > 1) else len(counter)
        min_count = min_count if (min_count and min_count > 1) else 0
        prune = set()
        for i, (word, freq) in enumerate(sorted(counter.items(), key=lambda x: x[1], reverse=True)):
            if i >= max_vocab_size or freq < min_count:
                prune.add(word)
        for word in prune:
            del counter[word]

    return counter


def class_assign(corpus: str, num_class: int, min_count: int):
    counter = freq_counter(corpus, min_count=min_count)
    words_with_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    words_sorted = [x[0] for x in words_with_freq]
    class_list = []
    chunk_size = len(words_sorted) // num_class + 1
    for i in range(0, len(words_sorted)):
        class_list.append(i // chunk_size)
    return zip(words_sorted, class_list)


def class_assign_by_pos(corpus: str, lang='zh', n_pos=2, max_vocab_size: int = None, min_count: int = None):  # TODO
    """ 依据POS信息为token做分类，由于POS标记存在错误，所以一个token可能有多个POS，使用频率最高的前n个POS由n_pos控制

    Args:
        corpus:     读取的文件
        lang:       语言，'en' for English and 'zh'表示中文
        n_pos:      使用频率最高的前n个POS作为每个token的分类标准
        max_vocab_size:  最大词表个数
        min_count:      最小词频
    """
    if lang == 'en':
        from nltk import pos_tag
        tagger = pos_tag
    elif lang == 'zh':
        import jieba.posseg as posseg
        tagger = posseg.cut
    else:
        raise ValueError("'zh' or 'en' only, got '{}'".format(lang))

    token_pos = defaultdict(lambda: Counter())

    # 统计词频的同时完成分词
    with open(corpus, encoding="utf8") as f:
        for line in tqdm(f, desc="Count & tag"):
            for token, pos in tagger(line):
                token_pos[token][pos] += 1

    # 根据词频或词表大小做筛选
    if max_vocab_size or min_count:
        removed = set()

        token_freq = [(tkn, sum(poss.values())) for tkn, poss in token_pos.items()]
        sorted(token_freq, key=lambda x: x[1], reverse=True)

        max_vocab_size = max_vocab_size if (max_vocab_size and max_vocab_size > 1) else len(token_pos)
        min_count = min_count if (min_count and min_count > 1) else 0

        for i, (token, cnt) in enumerate(token_freq):
            if i >= max_vocab_size or cnt < min_count:
                removed.add(token)

        for token in removed:
            del token_pos[token]

    # 分类
    classes = {}
    result = []
    for token, pos_cnt in token_pos.items():
        poss = sorted(pos_cnt.items(), key=lambda x: x[1], reverse=True)[:n_pos]
        poss = tuple(x[0] for x in poss)
        if poss not in classes:
            classes[poss] = len(classes)
        result.append((token, classes[poss]))

    return sorted(result, key=lambda x: x[1])


def class_assign_by_cilin(corpus: str, cilin: CiLin, level: int, max_vocab_size: int = None, min_count: int = None):
    """ 使用CiLin构建一棵语义分类树 """
    word_freq = freq_counter(corpus, max_vocab_size, min_count)
    code2word = defaultdict(set)

    hit, cnt = 0, 0
    monitor = tqdm(word_freq.items())
    for word, freq in monitor:
        if word not in cilin:
            code, _sim_words = cilin.find_similar_group(word, level)
            if len(_sim_words) == 0:
                code, _sim_words = cilin.guess_group(word, level)
        else:
            code = random.choice(list(cilin[word]))  # TODO：多义词分类问题，暂时使用随机挑选方式解决
            code = code[:level2len[level]]
            hit += 1
        cnt += 1
        monitor.set_postfix({"hit rate": hit / cnt})

        code2word[code].add(word)

    result = []
    for i, code in enumerate(sorted(code2word.keys())):
        result.extend((_w, i) for _w in sorted(code2word[code]))
    return result


def class_assign_by_word2vec(wv_path: str, vocab_path: str, n_clusters: int):
    import gensim
    import numpy as np
    from sklearn.cluster import k_means

    wv = gensim.models.keyedvectors.Word2VecKeyedVectors.load(wv_path)

    with open(vocab_path, encoding="utf8") as f:
        vocab = f.read().split()

    SPE_TOKEN = ("<s>", "</s>", "<unk>", "<pad>")
    vocab = [v in vocab if v not in SPE_TOKEN]
    vecs = np.array([wv[v] for v in vocab])
    _centroid, labels, _inertia, _n_iter = k_means(vecs, n_clusters=n_clusters, return_n_iter=True)

    classes = {i: [] for i in range(n_clusters)}
    for label, word in zip(labels, vocab):
        classes[label].append(word)

    result = [(t, 0) for t in SPE_TOKEN]
    for i in range(n_clusters):
        result.extend((w, i + 1) for w in classes[i])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 通用参数
    parser.add_argument('--data', type=str, default='./data/sms/train.txt', help="获得词表的语料")
    parser.add_argument("--w2v", type=str, help="w2v的保存位置")
    parser.add_argument("--method", type=str, choices=("cilin", "pos"), default="cilin", help="使用分词的方法")
    parser.add_argument('--max_vocab', type=int, default=None, help='最大词表数量')
    parser.add_argument('--min_count', type=int, default=0, help='最低词频')
    parser.add_argument('--save', type=str, default="./data/sms/vocab.c.txt", help='保存到什么位置')
    # 不同方法对应的参数
    parser.add_argument('--ncls', type=int, default=30, help='类个数')
    parser.add_argument("--level", "-l", type=int, choices=(1, 2, 3, 4, 5), default=4, help="词林层数")
    parser.add_argument('--lang', type=str, choices=("zh", "en"), default='zh', help="语料语言")
    parser.add_argument('--n_pos', type=int, default=2, help="使用几个POS标签作为类别")
    args = parser.parse_args()

    if args.method == "cilin":
        words_with_cls = class_assign_by_cilin(args.data, CiLin(), args.level,
                                               args.max_vocab, args.min_count)
    elif args.method == "pos":
        words_with_cls = class_assign_by_pos(args.data, args.lang, args.n_pos, args.max_vocab, args.min_count)
    elif args.method == "w2v":
        words_with_cls = class_assign_by_word2vec(args.w2v, args.data, args.ncls)
    else:
        words_with_cls = class_assign(args.data, args.ncls, args.min_count)

    with open(args.save, 'w', encoding="utf8") as f:
        lines = [f"{w} {c}" for w, c in words_with_cls]
        f.write("\n".join(lines))
