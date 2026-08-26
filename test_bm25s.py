import bm25s
corpus = ["hello world", "world of warcraft", "hello there"]
tokens = bm25s.tokenize(corpus)
bm = bm25s.BM25()
bm.index(tokens)
q_tokens = bm25s.tokenize(["hello world", "there"])
scores = bm.get_scores(q_tokens)
print("SHAPE", scores.shape)
