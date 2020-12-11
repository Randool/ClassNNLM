import logging
import os.path as path
import random
from collections import defaultdict
from functools import lru_cache
from itertools import product
from math import log
from typing import Callable, Iterable

len2level = {0: 0, 1: 1, 2: 2, 4: 3, 5: 4, 7: 5, 8: 6}
level2len = {v: k for k, v in len2level.items()}
step = (1, 2, 4, 5, 7, 8)


def route_split(code: str):
    """ Aa01A01= --> (A, Aa, Aa01, Aa01A, Aa01A01, Aa01A01=) """
    level = len2level[len(code)]
    return tuple(code[:step[i]] for i in range(level))


def LCS(code1: str, code2: str) -> str:
    """ LCS(abc, abd) = ab """
    i = 0
    for c1, c2 in zip(code1, code2):
        if c1 != c2:
            break
        i += 1
    if i == 3 or i == 6:
        i -= 1
    return code1[:i]


def _tree():
    """ 递归树 """
    return defaultdict(_tree)


class CiLin:
    def __init__(self, cilin_path: str = None):
        self.word2code = defaultdict(set)   # 词到code的映射，一对多
        self.code2word = defaultdict(set)   # code到词的映射，一对多
        self.code_cnt = defaultdict(int)    # 统计code下子节点树，用于计算相似度
        self.tree = _tree()  # 便于索引
        if cilin_path is None:
            cilin_path = path.join(path.split(__file__)[0], "cilin.txt")
        self._build(cilin_path)

    def _build(self, cilin_path: str):
        with open(cilin_path, encoding="utf8") as f:
            for line in f:
                code, *words = line.split()
                if len(code) == 8:
                    code = code[:-1]

                # 添加映射
                self.code2word[code].update(words)
                for w in words:
                    self.word2code[w].add(code)

                # 统计词频，顺便建树
                route = route_split(code)  # type: tuple
                ptr = self.tree
                for _code in route:
                    self.code_cnt[_code] += 1 #len(words)
                    ptr = ptr[_code]
                self.code_cnt[""] += 1 #len(words)

    @lru_cache(2048)
    def IC(self, c: str) -> float:
        """ information content """
        if len(c) == 0:
            return 0
        info = log(self.code_cnt[c] + 1, 2) / log(self.code_cnt[""], 2)
        return 1 - info

    @lru_cache(8192)
    def code_similarity(self, c1: str, c2: str) -> float:
        """ 两个节点的相似性，可替换为其他类型 """
        if c1 == c2:
            return 1
        lcs = LCS(c1, c2)
        # return 2 * self.IC(lcs) / (self.IC(c1) + self.IC(c2))
        return 1 - (self.IC(c1) + self.IC(c2) - 2 * self.IC(lcs)) / 2

    @lru_cache(32768)
    def word_similarity(self, w1: str, w2: str) -> float:
        """ 计算两个词相似性 """
        if w1 not in self.word2code:
            logging.warning(f"'{w1}' not in cilin")
            return 0
        if w2 not in self.word2code:
            logging.warning(f"'{w2}' not in cilin")
            return 0
        if w1 == w2:
            return 1
        codes1, codes2 = self.word2code[w1], self.word2code[w2]
        sims = [self.code_similarity(c1, c2) for c1, c2 in product(codes1, codes2)]
        return max(sims)

    def find_similar_group(self, words: Iterable, level: int = 5, score_func: Callable[[str, str], float] = None):
        """ 给定一堆words，在词林中找到类似的词群。支持自定义比较两个词相似性的函数——score_func

        Args:
            words:      一堆word
            level:      词林的层次
            score_func: 打分函数, score = score_func(w1,w2), None则调用默认的cilin相似度
        Returns:
            final_code (str):     词林的类别编码
            similar_words (set):   相似的词
        """
        words = [w for w in words if w in self]
        if level < 1:
            level = 1
        if level > 5:
            level = 5

        final_code, ptr = "", self.tree
        score_func = score_func if score_func else self.word_similarity

        for _ in range(level):
            best_score, next_code = 0, None

            for code in ptr.keys():
                c_words = self.code2word[code]

                scores = [score_func(w1, w2) for (w1, w2) in product(c_words, words)]
                avg_score = sum(scores) / len(scores) if len(scores) > 0 else 0
                # 根据字长加权score，更长的词的相似性更有话语权，但实验结果似乎不是很好
                # scores, weight = 0, 0
                # for w1, w2 in product(c_words, words):
                #     w = len(w1) + len(w2)
                #     scores += score_func(w1, w2) * w
                #     weight += w
                # avg_score = scores / weight if scores > 0 else 0

                if avg_score > best_score:
                    best_score = avg_score
                    next_code = code

            if next_code is None:  # 如果实在找不到合适的子类，那么随机分配
                next_code = random.choice(list(ptr.keys()))

            final_code, ptr = next_code, ptr[next_code]

        return final_code, self.code2word[final_code]

    def guess_group(self, word: str, level: int = 5, score_func: Callable[[str, str], float] = None):
        """ 给定一个OOV，使用拆词法猜测其可能属于的群体，支持自定义的比较两个词相似性的函数score_func

        Args:
            words:      一堆word
            level:      词林的层次
            score_func: 打分函数, score = score_func(w1,w2), None则调用默认的cilin相似度
        Returns:
            final_code (str):     词林的类别编码
            similar_words (set):   相似的词
        """
        sub_words = set()
        for _len in range(1, len(word)):  # 枚举所有可能的子词
            for i in range(len(word) - _len + 1):
                sub_words.add(word[i: i + _len])
        return self.find_similar_group(sub_words, level, score_func)

    def __contains__(self, word: str) -> bool:
        return word in self.word2code

    def __getitem__(self, word: str) -> set:
        if word not in self:
            return set()
        return self.word2code[word]


if __name__ == '__main__':
    _cilin = CiLin()

    def test_sim():
        print(_cilin.word_similarity("失迎", "失敬"))  # La05C02=  La05C01=
        print(_cilin.word_similarity("不等式", "等式"))  # Jb02A05@ Dc01C16#
        print(_cilin.word_similarity("不等式", "不等式"))  # Jb02A05@ Jb02A05@
        print(_cilin.word_similarity("苹果", "材料"))
        print(_cilin.word_similarity("苹果", "植物"))
        print(_cilin.find_similar_group(['菠萝', "苹果", '扁桃'], 4))
        print(_cilin.find_similar_group(['菠萝', "苹果", '扁桃'], 5))

    def test_level(name, ptr, level):
        if level > 0 and len(ptr.keys()) == 0:
            print(name, ptr)
        for k in ptr.keys():
            test_level(k, ptr[k], level - 1)

    def test_OOV(use_hownet=False):
        if use_hownet:
            import OpenHowNet
            hownet = OpenHowNet.HowNetDict()
            hownet.initialize_sememe_similarity_calculation()
            score_func = hownet.calculate_word_similarity
        else:
            score_func = None

        with open("../d.txt", encoding="utf8") as f:
            words = f.read().split()

        buf = []
        for word in words:
            _, sim_words = _cilin.guess_group(word, 5, score_func)
            line = f"{word}:\t{','.join(sim_words)}"
            print(line)
            buf.append(line)

        with open("../dd.txt", "w", encoding="utf8") as f:
            f.write("\n".join(buf))

    test_sim()
    #test_level("", _cilin.tree, 5)
    # test_OOV()
