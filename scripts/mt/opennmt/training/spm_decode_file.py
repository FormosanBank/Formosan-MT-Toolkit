import sys, sentencepiece as spm
if len(sys.argv) < 3:
    sys.exit("usage: spm_decode_file.py <spm.model> <pieces.txt>")
spp = spm.SentencePieceProcessor(model_file=sys.argv[1])
with open(sys.argv[2], "r", encoding="utf-8") as f:
    for line in f:
        print(spp.decode(line.strip().split()))