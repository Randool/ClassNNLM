import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(message)s", datefmt="%H:%M:%S")

"""
Word2Vec(sentences=None, corpus_file=None, size=100, alpha=0.025, window=5, 
         min_count=5, max_vocab_size=None, sample=1e-3, seed=1, workers=3,
         min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1,
         hashfxn=hash, iter=5, null_word=0, trim_rule=None, sorted_vocab=1,
         batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(), max_final_vocab=None)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-i", type=str, required=True, help="训练数据")
    parser.add_argument("--dim", "-d", type=int, default=128, help="维度（默认128）")
    parser.add_argument("--window", "-w", type=int, default=5, help="窗口宽度（默认5）")
    parser.add_argument("--max_vocab", "-M", type=int, help="最大词表")
    parser.add_argument("--min_count", "-m", type=int, default=1, help="最小词频（默认1）")
    parser.add_argument("--iter", type=int, default=10, help="训练迭代次数（默认10）")
    parser.add_argument("--worker", "-n", type=int, default=3, help="Worker数量（默认3）")
    parser.add_argument("--save", "-o", type=str, required=True, help="保存位置，格式gensim.models.keyedvectors.Word2VecKeyedVectors")
    args = parser.parse_args()

    from gensim.models import word2vec

    sentences = word2vec.LineSentence(args.data)

    model = word2vec.Word2Vec(
        sentences=sentences, size=args.dim, window=args.window,
        max_final_vocab=args.max_vocab, min_count=args.min_count,
        iter=args.iter, workers=args.workers, sg=1, negative=10
    )
    wv = model.wv
    logging.info(f"{len(wv.vocab)} tokens")

    wv.save(args.save)
    logging.info(f"Save to {args.save}")
