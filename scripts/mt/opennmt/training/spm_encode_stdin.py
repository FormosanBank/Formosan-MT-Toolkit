import sys, sentencepiece as spm
if len(sys.argv) < 2:
    sys.exit("usage: spm_encode_stdin.py <spm.model>")
spp = spm.SentencePieceProcessor(model_file=sys.argv[1])
for line in sys.stdin:
    print(" ".join(spp.encode(line.rstrip("\n"), out_type=str)))