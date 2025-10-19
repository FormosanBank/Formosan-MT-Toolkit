#!/usr/bin/env python3
import argparse, sentencepiece as spm, sys
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
spp = spm.SentencePieceProcessor(model_file=args.model)
with open(args.inp,"r",encoding="utf-8") as f, open(args.out,"w",encoding="utf-8") as g:
    for line in f:
        g.write(spp.decode(line.strip().split()) + "\n")
