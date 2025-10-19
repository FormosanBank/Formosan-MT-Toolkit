#!/usr/bin/env python3
import argparse, pathlib, sentencepiece as spm

def sanitize_ud_symbols(txt: str):
    parts = [s.strip() for s in txt.replace("\n","").split(",")]
    out, seen = [], set()
    for s in parts:
        if s and s not in seen:
            out.append(s); seen.add(s)
    return ",".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_dir", required=True, help="e.g., /path/to/ami_zh")
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--model_type", default="unigram", choices=["unigram","bpe"])
    ap.add_argument("--retrain", action="store_true", help="delete existing spm.model/vocab first")
    args = ap.parse_args()

    p = pathlib.Path(args.pair_dir)
    train_src = p/"raw/train/bi.src"
    train_tgt = p/"raw/train/bi.tgt"
    assert train_src.exists() and train_tgt.exists(), f"Missing {train_src} or {train_tgt}"

    if args.retrain:
        for f in [p/"spm.model", p/"spm.vocab"]:
            try: f.unlink()
            except FileNotFoundError: pass

    uds_file = p/"user_defined_symbols.txt"
    uds = ""
    if uds_file.exists():
        uds = sanitize_ud_symbols(uds_file.read_text(encoding="utf-8"))

    spm.SentencePieceTrainer.Train(
        input=",".join([str(train_src), str(train_tgt)]),
        model_prefix=str(p/"spm"),
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        normalization_rule_name="nmt_nfkc",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=uds  # <- clean, no blanks
        # Optional stabilizers for very small corpora:
        # hard_vocab_limit=False, byte_fallback=True
    )

    # Encode splits with Python API (no CLI needed)
    spp = spm.SentencePieceProcessor(model_file=str(p/"spm.model"))

    def encode(fin, fout):
        fout.parent.mkdir(parents=True, exist_ok=True)
        with open(fin, "r", encoding="utf-8") as f, open(fout, "w", encoding="utf-8") as g:
            for line in f:
                g.write(" ".join(spp.encode(line.strip(), out_type=str)) + "\n")

    for split in ["train", "valid", "test"]:
        encode(p/f"raw/{split}/bi.src", p/f"{split}/bi.spm.src")
        encode(p/f"raw/{split}/bi.tgt", p/f"{split}/bi.spm.tgt")

    print(f"[OK] Trained {p/'spm.model'} and encoded train/valid/test.")

if __name__ == "__main__":
    main()
