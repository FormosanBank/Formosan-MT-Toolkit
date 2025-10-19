#!/usr/bin/env python3
import argparse, csv, os, re, sys, pathlib
from collections import Counter

RESERVED_META = {"source","kindOf","dialect","row_type","split"}

def guess_form_col(header):
    low = [h.lower() for h in header]
    # first non-meta and not english/chinese
    for h, hl in zip(header, low):
        if hl in RESERVED_META or hl in {"english","en","chinese","zh"}:
            continue
        return h
    raise ValueError("Could not find Formosan text column.")

def get_ref_col(code, header):
    low = [h.lower() for h in header]
    if code == "en":
        for k in ("english","en"):
            if k in low: return header[low.index(k)]
    if code == "zh":
        for k in ("chinese","zh"):
            if k in low: return header[low.index(k)]
    raise ValueError(f"Could not find column for {code} in {header}")

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())

def split_norm(s):
    s = (s or "train").strip().lower()
    if s.startswith("val"): return "valid"
    if s == "validate": return "valid"
    if s in {"train","valid","test"}: return s
    # Unknown → default to train
    return "train"

def ensure_open(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="*_en_processed.csv or *_zh_processed.csv")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    csv_path = pathlib.Path(args.csv)
    m = re.match(r"([a-z]{3})_(en|zh)_processed\.csv$", csv_path.name)
    if not m:
        sys.exit(f"Filename must match <form>_(en|zh)_processed.csv, got: {csv_path.name}")
    form_code, other_code = m.group(1), m.group(2)
    pair = f"{form_code}_{other_code}"

    out_root = pathlib.Path(args.outdir) / pair
    # Make raw dirs
    for sub in ("raw/train","raw/valid","raw/test"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        form_col = guess_form_col(header)
        ref_col  = get_ref_col(other_code, header)

        # Build file handles up-front
        fw = {"train":{}, "valid":{}, "test":{}}

        def add_train_valid_handles(split):
            for src,tgt in ((form_code,other_code),(other_code,form_code)):
                fw[split][f"{src}2{tgt}.src"] = ensure_open(out_root/f"raw/{split}/{src}2{tgt}.src")
                fw[split][f"{src}2{tgt}.tgt"] = ensure_open(out_root/f"raw/{split}/{src}2{tgt}.tgt")
            fw[split]["bi.src"] = ensure_open(out_root/f"raw/{split}/bi.src")
            fw[split]["bi.tgt"] = ensure_open(out_root/f"raw/{split}/bi.tgt")

        def add_test_handles():
            for src,tgt in ((form_code,other_code),(other_code,form_code)):
                fw["test"][f"{src}2{tgt}.src"]     = ensure_open(out_root/f"raw/test/{src}2{tgt}.src")
                fw["test"][f"{src}2{tgt}.ref.{tgt}"] = ensure_open(out_root/f"raw/test/{src}2{tgt}.ref.{tgt}")
            fw["test"]["bi.src"] = ensure_open(out_root/"raw/test/bi.src")
            fw["test"]["bi.tgt"] = ensure_open(out_root/"raw/test/bi.tgt")

        add_train_valid_handles("train")
        add_train_valid_handles("valid")
        add_test_handles()

        def tag_for(tgt): return f"<2{tgt}>"

        counts = Counter()

        for row in reader:
            split = split_norm(row.get("split"))
            form_txt = norm(row.get(form_col,""))
            ref_txt  = norm(row.get(ref_col,""))
            if not form_txt or not ref_txt:
                continue

            if split in ("train","valid"):
                # Form -> Other
                print(form_txt, file=fw[split][f"{form_code}2{other_code}.src"])
                print(ref_txt,  file=fw[split][f"{form_code}2{other_code}.tgt"])
                # Other -> Form
                print(ref_txt,  file=fw[split][f"{other_code}2{form_code}.src"])
                print(form_txt, file=fw[split][f"{other_code}2{form_code}.tgt"])
                # Bi-directional with target tag on source
                print(f"{tag_for(other_code)} {form_txt}", file=fw[split]["bi.src"])
                print(ref_txt,                           file=fw[split]["bi.tgt"])
                print(f"{tag_for(form_code)} {ref_txt}", file=fw[split]["bi.src"])
                print(form_txt,                          file=fw[split]["bi.tgt"])

                counts[f"{form_code}2{other_code}_{split}"] += 1
                counts[f"{other_code}2{form_code}_{split}"] += 1

            elif split == "test":
                # Save paired test src and refs for BOTH directions + bi (for quick smoke tests)
                print(form_txt, file=fw["test"][f"{form_code}2{other_code}.src"])
                print(ref_txt,  file=fw["test"][f"{form_code}2{other_code}.ref.{other_code}"])
                print(ref_txt,  file=fw["test"][f"{other_code}2{form_code}.src"])
                print(form_txt, file=fw["test"][f"{other_code}2{form_code}.ref.{form_code}"])
                print(f"{tag_for(other_code)} {form_txt}", file=fw["test"]["bi.src"])
                print(ref_txt,                           file=fw["test"]["bi.tgt"])
                print(f"{tag_for(form_code)} {ref_txt}", file=fw["test"]["bi.src"])
                print(form_txt,                          file=fw["test"]["bi.tgt"])

                counts[f"{pair}_test_pairs"] += 1

        # Close all files
        for group in fw.values():
            for fh in group.values():
                fh.close()

    # Write user-defined symbols for SPM
    with open(out_root/"user_defined_symbols.txt","w",encoding="utf-8") as g:
        g.write(f"<2{form_code}>,<2{other_code}>\n")

    print(f"[OK] {pair} → wrote raw/ files")
    for k,v in sorted(counts.items()):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
