# 示例
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("t5_model.model")

tokens = sp.encode("Hello world!", out_type=str)
# ['▁Hello', '▁world', '!']