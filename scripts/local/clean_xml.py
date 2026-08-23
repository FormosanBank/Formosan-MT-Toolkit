#!/usr/bin/env python3
"""Prepare a derived XML copy for MT without mutating fetched source XML."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from pipeline_common import atomic_write_json, load_pipeline_config, sha256_file, stable_json_hash, utc_now
from qc_checkout import (
    DEFAULT_QC_CACHE,
    DEFAULT_QC_REVISION,
    QC_ORG,
    QC_REPO,
    SYNC_FILES,
    equivalent_lang_codes,
    resolve_qc_root,
)
from qc_inventory import (
    classify_translation_version_repairs,
    finalize_transform_inventory,
    mixed_content_signature,
    remove_transform_tags,
    snapshot_translation_versions,
    tag_transform_sources,
    write_repair_inventory,
    write_transform_inventory,
)
from qc_reporting import (
    QC_LOG_DIR,
    print_qc_rule_summary,
    run_captured_command,
    summarize_validator_findings,
)
from tqdm import tqdm
from xml_repairs import repair_mt_xml_structure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CONFIG = load_pipeline_config()
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

__all__ = [
    "SYNC_FILES",
    "audit_standard_tiers",
    "classify_translation_version_repairs",
    "ensure_standard_tiers",
    "finalize_transform_inventory",
    "tag_transform_sources",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean XML with a commit-pinned FormosanBank QC checkout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src-lang", required=True, help="Source ISO 639-3 code")
    parser.add_argument("--tgt-lang", help="Optional target-language eligibility check")
    parser.add_argument("--in-dir", help="Directory containing downloaded XML")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Derived XML directory. Must differ from --in-dir.",
    )
    parser.add_argument(
        "--formosanbank-path",
        type=Path,
        default=None,
        help="Explicit FormosanBank checkout. Its HEAD must equal --qc-revision.",
    )
    parser.add_argument(
        "--qc-revision",
        default=DEFAULT_QC_REVISION,
        help="Immutable FormosanBank commit containing the QC implementation.",
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=DEFAULT_QC_CACHE,
        help="Root for commit-addressed FormosanBank QC caches.",
    )
    parser.add_argument("--force-update", action="store_true", help="Re-verify and refresh the pinned QC cache.")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip FormosanBank hard validators. Intended only for local diagnosis.",
    )
    return parser.parse_args()


def xml_lang(elem: ET.Element) -> str:
    return (elem.attrib.get(XML_LANG) or elem.attrib.get("xml:lang") or "").strip().lower()


def validate_input_files(directory: Path, src_lang: str, tgt_lang: str | None) -> dict[str, int]:
    files = sorted(directory.rglob("*.xml"))
    if not files:
        raise SystemExit(f"No XML files found under {directory}")
    expected_source = src_lang.strip().lower()
    target_codes = equivalent_lang_codes(tgt_lang)
    counts: Counter[str] = Counter()
    errors: list[str] = []
    for path in files:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            errors.append(f"{path}: malformed XML ({exc})")
            counts["parse_errors"] += 1
            continue
        counts["xml_files"] += 1
        if xml_lang(root) != expected_source:
            errors.append(f"{path}: root xml:lang={xml_lang(root)!r}, expected {expected_source!r}")
            counts["source_language_mismatch"] += 1
        if target_codes:
            if any(xml_lang(transl) in target_codes for transl in root.iter("TRANSL")):
                counts["target_eligible_files"] += 1
            else:
                counts["target_ineligible_files"] += 1
    if errors:
        preview = "\n".join(f"  - {message}" for message in errors[:25])
        suffix = f"\n  ... and {len(errors) - 25} more" if len(errors) > 25 else ""
        raise SystemExit(f"Input XML validation failed:\n{preview}{suffix}")
    return dict(counts)


def validate_fetch_contract(directory: Path) -> dict[str, object]:
    manifest_path = directory / "_fetch_manifest.json"
    inventory_path = directory / "_fetch_inventory.jsonl"
    if not manifest_path.is_file() or not inventory_path.is_file():
        raise SystemExit(
            f"Missing fetch manifest/inventory under {directory}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Malformed fetch manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.get("complete") is not True:
        raise SystemExit(f"Fetch manifest is incomplete: {manifest_path}")
    inventory_hash = sha256_file(inventory_path)
    if inventory_hash != manifest.get("inventory_sha256"):
        raise SystemExit(
            f"Fetch inventory hash mismatch under {directory}"
        )
    kept: dict[str, str] = {}
    with inventory_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Malformed fetch inventory "
                    f"{inventory_path}:{line_number}: {exc}"
                ) from exc
            if row.get("status") == "kept":
                destination = str(row.get("destination") or "")
                expected_sha256 = str(row.get("sha256") or "")
                if not destination or len(expected_sha256) != 64:
                    raise SystemExit(
                        f"Incomplete kept fetch record at {inventory_path}:{line_number}"
                    )
                kept[destination] = expected_sha256
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*.xml")
    }
    expected = set(kept)
    if actual != expected:
        raise SystemExit(
            "Fetched XML set does not match its immutable inventory: "
            f"missing={sorted(expected - actual)[:10]}, "
            f"unexpected={sorted(actual - expected)[:10]}"
        )
    mismatched = [
        relative
        for relative, expected_sha256 in kept.items()
        if sha256_file(directory / relative) != expected_sha256
    ]
    if mismatched:
        raise SystemExit(
            "Fetched XML content does not match its immutable inventory: "
            f"mismatched={mismatched[:10]}"
        )
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "inventory_sha256": inventory_hash,
        "xml_files": len(actual),
        "xml_inventory_sha256": stable_json_hash(sorted(kept.items())),
    }


def prepare_working_copy(source_dir: Path, output_dir: Path) -> None:
    source = source_dir.resolve()
    output = output_dir.resolve()
    if source == output:
        raise SystemExit("--out-dir must differ from --in-dir; fetched XML is immutable")
    if source in output.parents or output in source.parents:
        raise SystemExit("--out-dir cannot contain or be contained by --in-dir")
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)


def run_dialect_completion(
    corpus_dir: Path,
    qc_root: Path,
    *,
    log_path: Path,
) -> tuple[
    dict[str, int],
    list[dict[str, object]],
    dict[str, object],
]:
    before: dict[str, str] = {}
    for xml_file in sorted(corpus_dir.rglob("*.xml")):
        root = ET.parse(xml_file).getroot()
        before[str(xml_file.relative_to(corpus_dir))] = (
            root.get("dialect") or ""
        ).strip()

    utility = qc_root / "QC" / "utilities" / "fix_dialects.py"
    cmd = [sys.executable, str(utility), "--path", str(corpus_dir)]
    log = run_captured_command(
        cmd,
        qc_root,
        log_path=log_path,
    )

    repairs: list[dict[str, object]] = []
    for xml_file in sorted(corpus_dir.rglob("*.xml")):
        relative = str(xml_file.relative_to(corpus_dir))
        root = ET.parse(xml_file).getroot()
        after = (root.get("dialect") or "").strip()
        if not after:
            raise SystemExit(
                f"Pinned dialect completion left TEXT/@dialect empty: {relative}"
            )
        if before[relative] != after:
            repairs.append(
                {
                    "repair": (
                        "normalize_dialect_alias"
                        if before[relative]
                        else "complete_missing_dialect"
                    ),
                    "xml_path": relative,
                    "before": before[relative],
                    "after": after,
                }
            )
    stats = {
        "files_scanned": len(before),
        "dialects_completed": sum(
            row["repair"] == "complete_missing_dialect"
            for row in repairs
        ),
        "dialects_normalized": sum(
            row["repair"] == "normalize_dialect_alias"
            for row in repairs
        ),
        "dialects_preserved": len(before) - len(repairs),
    }
    return stats, repairs, log


def run_translation_version_completion(
    corpus_dir: Path,
    qc_root: Path,
    *,
    log_path: Path,
) -> tuple[dict[str, int], list[dict[str, object]], dict[str, object]]:
    before = snapshot_translation_versions(corpus_dir)
    utility = (
        qc_root
        / "QC"
        / "utilities"
        / "fix_multiple_translations.py"
    )
    log = run_captured_command(
        [sys.executable, str(utility), "--path", str(corpus_dir)],
        qc_root,
        log_path=log_path,
    )
    repairs = classify_translation_version_repairs(
        before,
        snapshot_translation_versions(corpus_dir),
    )
    return (
        {
            "translations_scanned": len(before),
            "alternates_marked": len(repairs),
        },
        repairs,
        log,
    )


def copy_form(source: ET.Element, kind_of: str) -> ET.Element:
    copied = copy.deepcopy(source)
    copied.set("kindOf", kind_of)
    return copied


def ensure_standard_tiers(corpus_dir: Path) -> dict[str, int | str]:
    """Complete missing tiers while preserving every existing standard tier."""
    stats: Counter[str] = Counter()
    existing_before: dict[tuple[str, str, str], str] = {}
    existing_after: dict[tuple[str, str, str], str] = {}

    for xml_file in sorted(corpus_dir.rglob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        changed = False
        relative = str(xml_file.relative_to(corpus_dir))
        for parent in root.iter():
            if parent.tag not in {"S", "W", "M"}:
                continue
            forms = parent.findall("FORM")
            if not forms:
                continue
            originals = [form for form in forms if (form.get("kindOf") or "").strip().lower() == "original"]
            standards = [form for form in forms if (form.get("kindOf") or "").strip().lower() == "standard"]
            untyped = [form for form in forms if not (form.get("kindOf") or "").strip()]
            element_key = (relative, parent.tag, parent.get("id", ""))

            if len(originals) > 1 or len(standards) > 1:
                raise SystemExit(f"Duplicate original/standard FORM tier at {element_key}")
            if standards:
                existing_before[element_key] = mixed_content_signature(standards[0])
                stats["existing_standard"] += 1

            if not originals:
                if standards:
                    original = copy_form(standards[0], "original")
                    parent.insert(list(parent).index(standards[0]), original)
                    originals = [original]
                    stats["original_copied_from_standard"] += 1
                    changed = True
                elif len(untyped) == 1:
                    untyped[0].set("kindOf", "original")
                    originals = [untyped[0]]
                    stats["untyped_promoted_to_original"] += 1
                    changed = True
                else:
                    raise SystemExit(f"Cannot identify original FORM tier at {element_key}")

            if not standards:
                standard = copy_form(originals[0], "standard")
                insert_at = list(parent).index(originals[0]) + 1
                parent.insert(insert_at, standard)
                standards = [standard]
                stats["standard_copied_from_original"] += 1
                changed = True

            if element_key in existing_before:
                existing_after[element_key] = mixed_content_signature(standards[0])

        if changed:
            tree.write(xml_file, encoding="utf-8", xml_declaration=True)
            stats["files_changed"] += 1

    if existing_before != existing_after:
        changed_keys = sorted(key for key, value in existing_before.items() if existing_after.get(key) != value)
        raise SystemExit(
            "Tier completion modified existing standard tiers: " + ", ".join("/".join(key) for key in changed_keys[:10])
        )
    stats["preserved_standard_digest"] = stable_json_hash(
        [(*key, value) for key, value in sorted(existing_before.items())]
    )
    return dict(stats)


def audit_standard_tiers(corpus_dir: Path) -> dict[str, int | str]:
    stats: Counter[str] = Counter()
    signatures: list[tuple[str, str, str, str]] = []
    for xml_file in sorted(corpus_dir.rglob("*.xml")):
        root = ET.parse(xml_file).getroot()
        relative = str(xml_file.relative_to(corpus_dir))
        for parent in root.iter():
            if parent.tag not in {"S", "W", "M"}:
                continue
            standards = [
                form for form in parent.findall("FORM") if (form.get("kindOf") or "").strip().lower() == "standard"
            ]
            if len(standards) > 1:
                raise SystemExit(
                    f"Expected at most one standard FORM at "
                    f"{relative}:{parent.tag}:{parent.get('id', '')}"
                )
            if not standards:
                if parent.tag == "S" and parent.findall("AUDIO"):
                    stats["untranscribed_audio_sentences"] += 1
                else:
                    stats[
                        f"missing_{parent.tag.lower()}_standard_tiers"
                    ] += 1
                continue
            text = "".join(standards[0].itertext()).strip()
            if not text:
                if parent.tag == "S":
                    if standards[0].find("UNCLEAR") is not None:
                        stats["unclear_sentence_standard_tiers"] += 1
                    elif parent.findall("AUDIO"):
                        stats[
                            "untranscribed_audio_sentence_standard_tiers"
                        ] += 1
                    else:
                        stats["empty_source_sentences"] += 1
                stats[f"empty_{parent.tag.lower()}_standard_tiers"] += 1
            stats[f"{parent.tag.lower()}_standard_tiers"] += 1
            signatures.append((relative, parent.tag, parent.get("id", ""), mixed_content_signature(standards[0])))
    stats["standard_tier_digest"] = stable_json_hash(signatures)
    return dict(stats)


def run_qc_scripts(
    corpus_dir: Path,
    qc_root: Path,
    *,
    validate: bool,
) -> dict[str, object]:
    logs_dir = corpus_dir / QC_LOG_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    transform_sources = tag_transform_sources(corpus_dir)
    try:
        tier_completion = ensure_standard_tiers(corpus_dir)
        (
            translation_version_completion,
            translation_version_repairs,
            translation_version_log,
        ) = run_translation_version_completion(
            corpus_dir,
            qc_root,
            log_path=logs_dir / "fix_multiple_translations.log",
        )
        (
            dialect_completion,
            dialect_repairs,
            dialect_log,
        ) = run_dialect_completion(
            corpus_dir,
            qc_root,
            log_path=logs_dir / "fix_dialects.log",
        )
        (
            mt_structure_repair,
            structure_repairs,
            removal_dispositions,
        ) = repair_mt_xml_structure(corpus_dir)
        transform_rows = finalize_transform_inventory(
            corpus_dir,
            transform_sources,
            removal_dispositions,
        )
    except BaseException:
        remove_transform_tags(corpus_dir)
        raise
    transform_inventory = write_transform_inventory(
        corpus_dir,
        transform_rows,
    )
    repair_inventory = write_repair_inventory(
        corpus_dir,
        [
            *translation_version_repairs,
            *dialect_repairs,
            *structure_repairs,
        ],
    )
    standard_audit = audit_standard_tiers(corpus_dir)

    validators: list[dict[str, object]] = []
    if validate:
        with tqdm(
            total=2,
            desc="QC validate XML",
            unit="check",
            dynamic_ncols=True,
        ) as progress:
            for script_name in (
                "validate_xml.py",
                "validate_text.py",
            ):
                validator = (
                    qc_root / "QC" / "validation" / script_name
                )
                stem = Path(script_name).stem
                findings = corpus_dir / f"_qc_{stem}_findings.csv"
                log = run_captured_command(
                    [
                        sys.executable,
                        str(validator),
                        "--no-exit-on-hard",
                        "by_path",
                        "--path",
                        str(corpus_dir),
                        "--csv",
                        str(findings),
                    ],
                    qc_root,
                    log_path=logs_dir / f"{stem}.log",
                )
                validators.append(
                    {
                        "script": script_name,
                        "findings": findings.name,
                        "findings_sha256": sha256_file(findings),
                        "summary": summarize_validator_findings(
                            findings
                        ),
                        "log": log,
                    }
                )
                progress.update(1)
    return {
        "tier_completion": tier_completion,
        "translation_version_completion": (
            translation_version_completion
        ),
        "translation_version_log": translation_version_log,
        "semantic_text_cleaning": {
            "applied": False,
            "authority": "formosan-mt-standardization",
            "reason": (
                "FormosanBank source tiers are preserved; model text is "
                "derived by standardize_mt_corpus.py"
            ),
        },
        "dialect_completion": dialect_completion,
        "dialect_log": dialect_log,
        "mt_structure_repair": mt_structure_repair,
        "transform_inventory": transform_inventory,
        "repair_inventory": repair_inventory,
        "standard_audit": standard_audit,
        "validators": validators,
    }


def main() -> None:
    args = parse_args()
    source_dir = Path(args.in_dir or f"downloaded_{args.src_lang}").expanduser().resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {source_dir}")
    if args.out_dir is None:
        raise SystemExit("--out-dir is required so fetched source XML remains immutable")
    corpus_dir = args.out_dir.expanduser().resolve()

    fetch_snapshot_before = validate_fetch_contract(source_dir)
    prepare_working_copy(source_dir, corpus_dir)
    qc_root = resolve_qc_root(args)
    input_counts = validate_input_files(corpus_dir, args.src_lang, args.tgt_lang)
    qc_result = run_qc_scripts(corpus_dir, qc_root, validate=not args.skip_validation)
    fetch_snapshot_after = validate_fetch_contract(source_dir)
    if fetch_snapshot_before != fetch_snapshot_after:
        raise SystemExit("Fetched source snapshot changed during XML preparation")
    manifest = {
        "schema_version": 3,
        "pipeline_version": PIPELINE_CONFIG["pipeline_version"],
        "created_at": utc_now(),
        "source_language": args.src_lang,
        "target_language": args.tgt_lang,
        "source_dir": str(source_dir),
        "corpus_dir": str(corpus_dir),
        "formosanbank_qc": {
            "repository": f"{QC_ORG}/{QC_REPO}",
            "revision": args.qc_revision.lower(),
            "snapshot_path": str(qc_root),
            "snapshot_manifest_sha256": (
                sha256_file(qc_root / "QC_SNAPSHOT.json") if (qc_root / "QC_SNAPSHOT.json").is_file() else None
            ),
        },
        "input": input_counts,
        "fetch_snapshot": fetch_snapshot_before,
        "source_immutable": True,
        **qc_result,
        "complete": not args.skip_validation,
    }
    manifest_path = corpus_dir / "_qc_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print_qc_rule_summary(args.src_lang, qc_result)
    print(f"QC manifest: {manifest_path}")


if __name__ == "__main__":
    main()
