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

### 2. FormosanBank QC

`clean_xml.py` uses one commit-pinned FormosanBank QC snapshot. Existing
`kindOf="standard"` tiers are never replaced with originals. Missing standards
are derived only where required, then processed by the pinned canonical
cleaner. Hard XML and text validators always run in production.

Before validation, the toolkit invokes the pinned FormosanBank dialect utility
to complete missing `TEXT/@dialect` values. It also applies a narrow MT repair
policy: source-empty lexical units are removed, punctuation-only text outside
typed XML fields is removed, and unreferenced duplicate IDs are deterministically
disambiguated. Units carrying FormosanBank hard annotation markers such as
`V129` asterisks are excluded wholesale rather than rewriting their standard
text. Boundary whitespace left by the pinned cleaner in direct-text `FORM`
elements is stripped without altering internal spacing. W/M units with
metalinguistic slash or parenthetical variants are excluded under `V121`, and
sentence standards containing the null/elision symbol are excluded under
`V120`. Forbidden zero-width characters are removed under `V131`. Empty
sentence units are quarantined wholesale, without falling back to their
`original` tier. Any empty sentence that survives this repair, substantive
untyped content, and referenced duplicate IDs remain hard failures. All repairs
are written to
`_qc_repair_inventory.jsonl`, and validator findings are stored beside the
cleaned XML rather than in the shared QC checkout.

Before QC, every S/W/M element receives a temporary transform ID. After QC the
temporary attribute is removed and a sidecar records whether the standard was
provided or derived, original/standard hashes before QC, the final standard
hash, source and final XML IDs, removal disposition, and stable final element
locator.

### 3. Extraction

`make_corpus.py` refuses XML not accounted for by the immutable fetch
inventory, verifies fetch/QC sidecar hashes, and extracts only
`FORM kindOf="standard"`. Stable row IDs include the XML element index, so
files with missing or repeated XML IDs remain traceable. Rows retain raw
original, post-QC standard, repository commit, QC commit, dialect, structural
type, target metadata, and transform hashes.

### 4. MT Filtering

`filter_split_corpus.py` performs conservative NFC, control-character,
HTML-entity, and whitespace normalization. It does not run English Moses rules
on Formosan/Chinese, NFKC the model text, or delete parenthetical spans.
Language/script, redaction, identity, markup, repetition, and broad fertility
checks quarantine or reject questionable rows. Exact pairs are deduplicated,
and every input row is conserved as accepted, rejected, quarantined, or
deduplicated in a ledger. This stage never assigns data splits.

### 5. Aggregation And Pivoting

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

### 6. Hard Splitting

`formosan_mt_experiments/scripts/build_experiment_splits.py` ignores legacy
pairwise split labels and builds the model-facing split. It:

1. deduplicates canonical source-target pairs;
2. locks lexical, morpheme, and synthetic rows to training;
3. forms connected components through normalized and punctuation/spacing
   skeleton source and target keys;
4. holds out complete human source documents where enough documents exist;
5. uses declared human group-level fallbacks for small languages;
6. removes one-edit and character 4-gram Jaccard conflicts on both sides across
   train/test, train/validation, and test/validation;
7. fails if any language has less than 7.5% test or 2.5% validation against the
   complete deduplicated corpus denominator.

The final denominator includes human, synthetic, sentence, and lexical rows.
This prevents pivot growth from making evaluation statistically negligible.

### 7. Independent Validation

`validate_experiment.py` independently recomputes provenance, split ratios,
document overlap, normalized/skeleton overlap, one-edit conflicts, and exact
character n-gram conflicts from the emitted CSV. It requires human sentence
evaluation and standard-tier provenance.

`audit_corpus_exposure.py` then runs TAME-MT 0.2.2 with exact native retrieval
in both MT directions. It reports SourceExposure, TargetExposure,
PairLeakTopK, exact overlap, and translation-memory baselines by split and
language. Exact overlap and exposure at 0.95 must be zero; 0.70 and 0.85 remain
diagnostics. Exposure is evidence of similarity risk, not proof of
memorization.

### 8. NLLB Training

`setup_tokenizer_sweep.py` and `setup_formosan_nllb200.py` learn the auxiliary
8k SPM from standard-tier Formosan training text only. They load a pinned
NLLB-200 revision, realign every shared embedding by token identity after the
SentencePiece ID shift, initialize new pieces from old subpieces, seed new
Formosan language IDs, add train-derived metadata tags, and hash every setup
artifact.

`train_directional_nllb.py` verifies the corpus, independent validation, setup,
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

## Data Products

| Product | Lifetime | Versioned |
|---|---|---|
| Downloaded XML and raw/filtered rebuild CSVs | Regenerable | No |
| DeepL JSONL response caches | Expensive source artifact | No; checksum and back up |
| Named final no-Bible corpora | Current experiment input | No; manifest/checksum yes |
| Split/validation/exposure reports | Current provenance | Packaged beside final corpus |
| Tokenizers, checkpoints, predictions | Cluster experiment output | No |
| Experiment configuration and job graph | Reproducibility record | Yes |

`pivot_corpora_final/provenance/bundle_manifest.json` is the portable training
contract. Historical model implementations and generated formats remain in Git
history rather than the active tree.
