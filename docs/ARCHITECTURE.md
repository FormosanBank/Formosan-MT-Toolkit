# Pipeline Architecture

## Supported Workflow

`scripts/local/build_mt_corpus.py` is the single orchestration boundary. The
top-level `build_corpora.sh` only selects the repository virtualenv and forwards
arguments. Individual stage scripts remain callable for debugging, but a
production corpus should come from the orchestrator so paths, cleanup, pivot
cache reuse, hard splitting, validation, and checksummed manifests stay
consistent.

### 1. Acquisition

`fetch_xml.py` enumerates repositories visible through GitHub, resolves each
branch to a commit, rejects truncated trees, and verifies every downloaded blob
against its Git object ID. Repository, tree, download, checksum, and parse
failures make the fetch manifest incomplete. Public mode restricts input to the
public FormosanBank corpus tree. Private mode means every eligible repository
visible to the token.

`--exclude-bible` excludes one exact repository/corpus root:
`Formosan-Taiwan-Bible-Society-Bibles`. It is not a fuzzy path-name filter.

### 2. Source Selection And Structural QC

`clean_xml.py` verifies the fetched snapshot, creates a separate
`prepared_<lang>` copy, and uses one commit-pinned FormosanBank QC snapshot.
The downloaded XML remains byte-for-byte immutable. For every S/W/M unit it
selects a nonempty supplied `kindOf="standard"` tier first, then a nonempty
original, then one nonempty untyped `FORM`. Existing standards are never
replaced. Missing tiers are completed only in the prepared copy. Hard XML and
text validators report findings without making semantic text decisions.

Before validation, the toolkit invokes the pinned FormosanBank dialect utility
to complete missing `TEXT/@dialect` values. It also applies a narrow MT repair
policy: punctuation-only text outside typed XML fields is removed, and
unreferenced duplicate IDs are deterministically disambiguated. XML units,
transcription tiers, and audio metadata are never deleted to satisfy MT
validation. Boundary whitespace in direct-text `FORM` elements is stripped
without altering internal spacing. Parentheses and slashes
in W/M forms are preserved because they may encode valid morphological or
lexical notation; the QC layer reports them only as review diagnostics.
Empty selected source tiers are recorded and skipped during standardization.
Null/elision and annotation notation remain in XML and are handled by the
separate MT standardizer rather than by structural QC.
Forbidden zero-width characters are removed under `V131`. Validator findings,
including invalid audio spans, are logged without rewriting the source record
or aborting corpus construction. Substantive untyped content and referenced
duplicate IDs remain hard structural failures. All repairs are written to
`_qc_repair_inventory.jsonl`, and validator findings are stored beside the
cleaned XML rather than in the shared QC checkout.

Pinned QC subprocess output is captured under `_qc_logs/`. Normal terminal
output uses file/stage progress bars followed by rule-level counts for tier
completion, repaired XML units, standardization outcomes, and validator
findings. Pair filtering likewise reports every
normalization, rejection, quarantine, and deduplication rule; the same counts
are stored in the corresponding manifests and filter report.

Before QC, every S/W/M element receives a temporary transform ID. After QC the
temporary attribute is removed and a sidecar records whether the standard was
provided or derived, original/standard hashes before QC, the final standard
hash, source and final XML IDs, removal disposition, and stable final element
locator. `_qc_transform_inventory.jsonl` records one source-selection outcome
for every unit, including untranscribed audio and unclear source text.

### 3. MT Standardization

`standardize_mt_corpus.py` consumes the source-selection ledger and applies
`config/mt_standardization.json`. This transformation never writes model text
back to XML. It emits one `_mt_standard_inventory.jsonl` record per source
unit with raw original, selected source standard, model standard, ordered
rules, confidence, disposition, evaluation eligibility, profile ID/hash, and
text hashes. Rules are NFC-based, source-aware, unit-aware, deterministic, and
idempotence checked. Unresolved notation is quarantined; unclear or empty
source units are ineligible; ambiguous canonicalization is train-only unless a
reviewed source override explicitly permits evaluation.

### 4. Extraction

`make_corpus.py` refuses XML not accounted for by the immutable fetch
inventory, verifies fetch/QC/MT-standard sidecar hashes, and extracts only
accepted `formosan_mt_standard` records. `formosan_sentence` is an exact alias
for that field, while archival original and source-standard values remain
separate. Stable row IDs include the XML element index, so
files with missing or repeated XML IDs remain traceable. Rows retain raw
original, post-QC standard, repository commit, QC commit, dialect, structural
type, target metadata, and transform hashes.

### 5. MT Filtering

`filter_split_corpus.py` performs conservative NFC, control-character,
HTML-entity, and whitespace normalization. It does not run English Moses rules
on Formosan/Chinese, NFKC the model text, or delete parenthetical spans.
Language/script, redaction, identity, markup, repetition, and broad fertility
checks quarantine or reject questionable rows. Exact pairs are deduplicated,
and every input row is conserved as accepted, rejected, quarantined, or
deduplicated in a ledger. This stage never assigns data splits.

### 6. Aggregation And Pivoting

The orchestrator combines all pairwise rows while retaining provenance and
`row_type`. `scripts/local/pivot.py` fills missing English or
Chinese targets, rotates all configured DeepL keys, batches within API limits,
backs off on transient errors, and writes each successful response immediately
to a content-keyed JSONL cache. Rebuilds read shared and build-local caches
before spending quota.

Pivot rows retain provider, direction, source text, origin, cache key, detected
language, and inherited XML provenance. Responses pass the same script,
identity, repetition, placeholder, and fertility checks as human rows.
Unresolved candidates, invalid cache records, row errors, quota exhaustion, or
a stop reason prevent output promotion and return a nonzero status.

### 7. Hard Splitting

`formosan_mt_experiments/scripts/build_experiment_splits.py` ignores legacy
pairwise split labels and builds the model-facing split. It:

1. deduplicates canonical source-target pairs;
2. locks lexical, morpheme, synthetic, ambiguous-normalization, and
   MT-evaluation-ineligible rows out of evaluation;
3. prefers clean, non-easy human sentences meeting the configured token floors,
   expanding to all clean human sentences only where needed to reach the
   per-language evaluation floor, and computes normalized plus
   punctuation/spacing-skeleton keys;
4. holds out complete source documents with an exact subset allocator that
   reserves both per-language evaluation floors;
5. keeps coarse documents intact and permits evaluation-size overshoot for
   languages where exact 90/2.5/7.5 proportions are impossible;
6. removes one-edit and character 4-gram Jaccard conflicts on both sides across
   train/test and train/validation globally; test/validation separation is
   target-language-task conditioned, with stricter cross-language reuse retained
   as a diagnostic; conflicting training rows are excluded after the human
   benchmark is selected, rather than shrinking the evaluation set;
7. fails if any language has less than 7.5% test or 2.5% validation against the
   complete deduplicated corpus denominator.

The final denominator includes human, synthetic, sentence, and lexical rows.
This prevents pivot growth from making evaluation statistically negligible.

### 8. Independent Validation

`validate_experiment.py` independently verifies the MT-standard profile and
alias contract, then recomputes provenance, split ratios,
document overlap, normalized/skeleton overlap, one-edit conflicts, and exact
character n-gram conflicts from the emitted CSV. It requires human sentence
evaluation and MT-standard provenance.

`audit_corpus_exposure.py` then runs TAME-MT 0.2.2 with exact native retrieval
in both MT directions. It reports SourceExposure, TargetExposure,
PairLeakTopK, exact overlap, and translation-memory baselines by split and
language. Retrieval is conditioned on `lang_code`, matching the explicit
language-control tag used by each multilingual model. It therefore rejects
within-language exposure; cross-language reuse remains a separate diagnostic
in independent split validation. Exact overlap and exposure at 0.95 must be
zero; 0.70 and 0.85 remain diagnostics. Exposure is evidence of similarity
risk, not proof of memorization.

### 9. NLLB And MADLAD Training

`setup_tokenizer_sweep.py` and `setup_formosan_nllb200.py` learn the auxiliary
8k SPM from V3 MT-standard Formosan training text only. They load a pinned
NLLB-200 revision, realign every shared embedding by token identity after the
SentencePiece ID shift, initialize new pieces from old subpieces, seed new
Formosan language IDs, add train-derived metadata tags, and hash every setup
artifact.

`train_directional.py` verifies the corpus, independent validation, setup,
profile, and file hashes before training. It trains one direction per
checkpoint with source-bucket weighting and language-temperature sampling,
performs fixed human per-language generation validation, selects best by
chrF2, supports early stopping, and binds every resume/checkpoint to an
immutable run contract.

`evaluate_directional.py` evaluates the entire human test split. Default
metadata tags are the headline score; oracle source/dialect tags are a separate
diagnostic. Reports include BLEU, chrF2, TER, sacreBLEU signatures, stratified
bootstrap confidence intervals, exact/empty/length diagnostics, and
language/source/dialect/length slices.

NLLB and MADLAD share `train_directional.py`, `evaluate_directional.py`, and
the same V3 corpus gate. `model_backends.py` owns only the family-specific
tokenizer, prompt, decoder-start, and target-selection behavior. Every setup,
run, checkpoint, and evaluation contract records the exact MT-standard profile
ID and SHA-256. Published Formosan-source models include the same normalizer
and profile used during corpus construction.

## Data Products

| Product | Lifetime | Versioned |
|---|---|---|
| Downloaded XML, prepared XML, and raw/filtered rebuild CSVs | Regenerable | No |
| DeepL JSONL response caches | Expensive source artifact | No; checksum and back up |
| Named final no-Bible corpora | Current experiment input | No; manifest/checksum yes |
| Split/validation/exposure reports | Current provenance | Packaged beside final corpus |
| Tokenizers, checkpoints, predictions | Cluster experiment output | No |
| Experiment configuration and job graph | Reproducibility record | Yes |

`pivot_corpora_final/provenance/bundle_manifest.json` is the portable training
contract. Historical model implementations and generated formats remain in Git
history rather than the active tree.
