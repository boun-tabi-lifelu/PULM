# UniRef50 (2026_02) SQLite Databases — Build & Schema Reference

**Release:** UniRef50, UniProt release `2026_02`
**Root directory:** `/cta/share/users/uniprot/uniref/uniref50_2026_02/`
**Source files:** `uniref50.xml`, `uniref50.fasta`

---

## 1. Overview

Four SQLite databases are built from the two UniRef50 source files. The XML file is the
primary source (it is the only one that carries full cluster membership); the FASTA file is
parsed independently and used purely as a correctness cross-check.

| Database | Built from | Built by | Purpose |
|---|---|---|---|
| `uniref50_representatives.db` | `uniref50.xml` | `uniref50_xml_to_db.py` | **Main working DB.** One row per cluster (representative sequence) + the pLM/tokenizer training splits |
| `uniref50_representatives_fasta.db` | `uniref50.fasta` | `uniref50_fasta_to_db.py` | Independent rebuild of the same table, used to cross-check the XML parse |
| `uniref50_members.db` | `uniref50.xml` | `uniref50_xml_to_db_all.py` | One row per **cluster member** (all ~244 M members, no sequences) |
| `uniref50_human.db` | `uniref50_members.db` | `uniref50_human_v1.py`, `uniref50_human_v2.py` | Human (taxon 9606) subsets, incl. the isoform-free "distilled" sets |

Derived tables inside `uniref50_representatives.db` are produced by
`prepare_uniref50_splits.py` (train/validation) and `create_tokenizer_train_data.py`
(tokenizer corpus).

Build order:

```
uniref50.xml ──> uniref50_xml_to_db.py     ──> uniref50_representatives.db
             │                                     │
             │                                     ├─ prepare_uniref50_splits.py     ──> plm_train, plm_validation
             │                                     └─ create_tokenizer_train_data.py ──> tokenizer_train
             │
             └──> uniref50_xml_to_db_all.py ──> uniref50_members.db
                                                   │
                                                   └─ uniref50_human_v{1,2}.py ──> uniref50_human.db

uniref50.fasta ──> uniref50_fasta_to_db.py ──> uniref50_representatives_fasta.db   (cross-check only)
```

---

## 2. `uniref50_representatives.db`

`/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50_representatives.db`

Tables: `uniref50_representatives`, `plm_train`, `plm_validation`, `tokenizer_train`

### 2.1 `uniref50_representatives` — 38,794,121 rows

One row per UniRef50 cluster, holding the metadata of the cluster plus the **representative
member's sequence**. Non-representative members are *not* in this table (see
`uniref50_members.db`).

| Column | Type | Source in XML | Notes |
|---|---|---|---|
| `entry_id` | TEXT | `<entry id=...>` | e.g. `UniRef50_A0ACG4HM60`. Unique index built after load |
| `cluster_name` | TEXT | `<name>` | `"Cluster: "` prefix stripped |
| `member_count` | INTEGER | property `member count` | Number of sequences in the cluster |
| `common_taxon_name` | TEXT | property `common taxon` | Lowest common taxon of the whole cluster |
| `common_taxon_id` | TEXT | property `common taxon ID` | NCBI taxid of the above |
| `representative_member_id` | TEXT | `representativeMember/dbReference@id` | e.g. `A0ACG4HM60_9NOCA` |
| `sequence_length` | INTEGER | `<sequence length=...>` | As declared by UniProt |
| `sequence` | TEXT | `<sequence>` text | Newlines stripped (UniRef wraps long sequences) |

**Build notes (`uniref50_xml_to_db.py`)**

- Streaming `iterparse` over the XML; `lxml` is used when available (~2–3× faster than
  `xml.etree`), with automatic fallback to the stdlib parser.
- Processed elements are cleared (and, under `lxml`, preceding siblings deleted) so memory
  stays flat over the ~38.8 M entries.
- Parsing stops at `<representativeMember>` within each entry, since the DTD places it
  before all `<member>` elements — everything needed for this table is read by then.
- Bulk-load PRAGMAs during the load: `journal_mode=OFF`, `synchronous=OFF`,
  `temp_store=MEMORY`, `cache_size=-400000` (~400 MB), `page_size=8192`.
- **No `PRIMARY KEY` at load time on purpose.** `entry_id` is a random-looking TEXT key, so
  maintaining a unique index during insertion would turn every insert into a random
  read/write in a multi-GB B-tree — costly on a shared filesystem. A `UNIQUE INDEX` on
  `entry_id` is created afterwards, then durability PRAGMAs are restored
  (`journal_mode=DELETE`, `synchronous=FULL`).
- Inserts are batched at 50,000 rows per commit.

### 2.2 `plm_train` — 38,357,530 rows / `plm_validation` — 193,971 rows

Homology-reduced train/validation splits for (continual) pLM pretraining, replicating the
ESM-2 / Meier et al. data-handling procedure. Both tables have the same minimal schema:

| Column | Type |
|---|---|
| `entry_id` | TEXT NOT NULL |
| `sequence` | TEXT NOT NULL |

Both are indexed on `entry_id` (`idx_plm_train_entry_id`, `idx_plm_validation_entry_id`).

**Build notes (`prepare_uniref50_splits.py`)**

1. **Partition.** A random `--val_frac = 0.005` (0.5%) of `uniref50_representatives` rows is
   held out as validation, seeded with `--seed 42`:
   `round(38,794,121 × 0.005) = 193,971` sequences. The remaining **38,600,150** form the
   pre-reduction train set. Selection is done on `rowid` without materialising the table;
   both splits are streamed out to temporary FASTA files in a single pass.
2. **Homology reduction.** MMseqs2 search with **query = train, target = validation**:
   `--min-seq-id 0.5 --alignment-mode 3 --max-seqs 300 -s 7 -c 0.8 --cov-mode 0`.
   Every train sequence hitting any validation sequence at **≥ 50% identity** is dropped.
   GPU mode is the default (padded target DB via `makepaddedseqdb`, `search --gpu 1`, device
   pinned with `CUDA_VISIBLE_DEVICES`); `--no_gpu` falls back to CPU `easy-search`.
3. **Persist.** Both splits are written back into the *same* database from the temporary
   FASTA files, then indexed. Temporary FASTA/MMseqs2 files are cleaned up.

**Reduction accounting:** 38,600,150 − 38,357,530 = **242,620 train sequences removed**
(0.63% of the pre-reduction train set) as homologous to the validation set.
Validation is never pruned: 193,971 = 38,794,121 + 0 − 38,600,150 exactly.

### 2.3 `tokenizer_train` — 2,000,000 rows

Corpus used for training/fitting the tokenizers (AA, BPE, PUMA, …). Schema:
`entry_id` TEXT, `sequence` TEXT; indexed on `entry_id` (`idx_tokenizer_entry`).

**Build notes (`create_tokenizer_train_data.py`)** — sampled **from `plm_train`**, so it is
disjoint from `plm_validation` by construction:

1. **Oversample.** 2,500,000 random `rowid`s (1.25 × target) are drawn from `plm_train` and
   fetched in 100k-row chunks, to absorb rowid gaps and rows lost in the next step.
2. **Ambiguity filter.** A sequence is kept only if it contains **at most one** occurrence of
   *each* of `X`, `B`, `U`, `Z`.
3. **Trim.** Exactly 2,000,000 rows are sampled from the survivors (`random_state=42`).
4. **Length capping.** Sequences longer than 3,000 residues are replaced by a **single
   contiguous 3,000-residue window** starting at a uniformly random valid offset. Sequences
   ≤ 3,000 are kept intact. Note that `entry_id` is preserved, so a row here may hold a
   *fragment* of the full representative sequence.

---

## 3. `uniref50_representatives_fasta.db` (cross-check)

`/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50_representatives_fasta.db`

Single table `uniref50_representatives`, **38,794,121 rows** — identical schema and identical
row count to the XML-derived table, which is the point: it validates the XML parse.

**Build notes (`uniref50_fasta_to_db.py`)**

- Streams `uniref50.fasta` (transparently handling `.gz`), buffering the header + sequence
  lines of each record and flushing them as one row.
- Header regex covers the UniRef50 definition line
  `>ID ClusterName n=Members Tax=Taxon TaxID=TaxonID RepID=RepresentativeMember`, with
  `TaxID=` optional (older releases omit it). Headers that fail the regex still yield a row
  with at least the `entry_id`, and are counted as malformed rather than silently dropped.
- Here `entry_id` **is** declared `PRIMARY KEY` and inserts use `INSERT OR IGNORE`
  (the FASTA file is small enough that this is not a bottleneck).
- `sequence_length` is **computed** as `len(sequence)`, whereas the XML version takes the
  declared `<sequence length=...>` attribute — a useful additional consistency check.
- Post-load indexes on `representative_member_id`, `common_taxon_id`, `sequence_length`.

**Field-level correspondence to the XML table:** `cluster_name` ↔ FASTA `ClusterName`,
`member_count` ↔ `n=`, `common_taxon_name` ↔ `Tax=`, `common_taxon_id` ↔ `TaxID=`,
`representative_member_id` ↔ `RepID=`.

---

## 4. `uniref50_members.db`

`/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50_members.db`

Single table `uniref50_members` — **243,725,749 rows** (≈ 6.3 members per cluster across the
38,794,121 clusters). **No sequences** here; this table is the membership/taxonomy map.

| Column | Type | Notes |
|---|---|---|
| `uniprot_id` | TEXT | `dbReference@id`: UniProtKB ID (`A0ACG4HM60_9NOCA`), an isoform ID (`Q8WZ42-8`), or a UniParc ID for UniParc-only members |
| `uniprot_accession` | TEXT | property `UniProtKB accession`. **NULL** for UniParc-only members |
| `uniparc_id` | TEXT | property `UniParc ID`; for `type="UniParc ID"` references the element id itself is used |
| `is_representative` | INTEGER | `1` for `<representativeMember>`, `0` for `<member>` |
| `representative_member` | TEXT | Cluster id, e.g. `UniRef50_Q8WZ42` (i.e. the FK to `uniref50_representatives.entry_id`) |
| `taxon_id` | TEXT | **Member-level** NCBI taxid (`9606` = human) |
| `common_taxon_id` | TEXT | **Cluster-level** common taxid, identical for every row of a cluster |

**Build notes (`uniref50_xml_to_db_all.py`)**

- Same streaming `iterparse` strategy (stdlib `ElementTree`, `root.clear()` after each entry),
  batch size 50,000, same bulk-load PRAGMAs.
- **No `PRIMARY KEY` / `UNIQUE` constraint on `uniprot_id`** — a member could in principle
  appear more than once, and a unique constraint would silently discard rows. Indexes on
  `uniprot_id`, `uniprot_accession`, `uniparc_id` and `representative_member` are built after
  the load.
- ⚠️ Because this table has no uniqueness constraint, re-running the script against an
  existing DB **appends** rather than replaces. Delete the file first when rebuilding.

---

## 5. `uniref50_human.db`

`/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50_human.db`

All three tables share the schema of `uniref50_members` (7 columns, see §4) and are produced
by a straight `pandas` pull of `SELECT * FROM uniref50_members WHERE taxon_id='9606'`.

| Table | Rows | Isoform policy | Script |
|---|---|---|---|
| `uniref50_human_all` | 264,465 | none — every human member row | both |
| `uniref50_human_distilled` | 83,748 | **all isoforms dropped** | `uniref50_human_v1.py` |
| `uniref50_human_distilled_v2` | 87,391 | isoforms kept **only if representative** | `uniref50_human_v2.py` |

### 5.1 Distillation logic (shared by both versions)

Starting from `uniref50_human_all`:

1. Drop rows with a NULL `uniprot_id`.
2. **Isoform filter** (the only difference between v1 and v2 — see below).
3. Keep every human **representative** row (`is_representative == 1`).
4. From the human **non-representative** rows, keep only those whose cluster representative is
   *not itself* in the human representative set (i.e. clusters represented by a non-human
   protein), and de-duplicate on `representative_member` so **at most one human sequence per
   UniRef50 cluster** survives.
5. Concatenate 3 + 4 and write to the target table.

The result is a non-redundant human proteome at 50% identity: one human protein per UniRef50
cluster that contains any human member.

### 5.2 v1 vs v2

- **v1 (`uniref50_human_distilled`)** — `uniprot_id` containing `-` is treated as an isoform
  and removed unconditionally. This matches the **original PUMA paper**, and is the version we
  use.
- **v2 (`uniref50_human_distilled_v2`)** — the isoform filter is relaxed with an OR clause:
  an isoform is retained if `is_representative == 1`. This recovers **3,643** additional rows
  (87,391 − 83,748): isoforms that UniRef50 itself elected as the cluster representative
  (e.g. `Q8WZ42-9` representing `UniRef50_Q8WZ42-9`), and which v1 discards even though no
  canonical human protein represents those clusters.

### 5.3 Null pattern

`uniprot_accession` is the only nullable column: 79,636 / 83,748 non-null in v1 and
83,570 / 87,391 in v2, i.e. ~4.1 k / ~3.8 k rows are **UniParc-only** members carrying no
UniProtKB accession. All other columns are fully populated.

---

## 6. Cross-cutting notes and caveats

- **Cross-check status.** XML- and FASTA-derived representative tables agree on row count
  (38,794,121). Any deeper verification (per-`entry_id` sequence equality,
  `sequence_length` declared vs. computed) can be run as a join across the two DBs.
- **Join keys.** `uniref50_representatives.entry_id` ↔ `uniref50_members.representative_member`
  ↔ `uniref50_human_*.representative_member`. Sequences live *only* in
  `uniref50_representatives`, so any human-subset analysis needs this join to obtain residues.
- **Two different taxon columns.** `taxon_id` is the member's own organism;
  `common_taxon_id` is the cluster's lowest common ancestor and can be far above species level
  (e.g. `7742` = Vertebrata for the titin cluster). Filter on `taxon_id` for "human proteins".
- **Split leakage.** `plm_train` / `plm_validation` are homology-reduced at 50% identity;
  `tokenizer_train` is drawn from `plm_train` only. `uniref50_human_distilled` is **not**
  excluded from `plm_train` — if a downstream human benchmark must be held out, that has to be
  done explicitly.
- **Truncation in `tokenizer_train`.** Rows are capped at 3,000 residues; do not treat them as
  full-length sequences.
- **Reproducibility.** Fixed seeds: `--seed 42` for the train/val partition,
  `random_state=42` for the tokenizer subsample. The 3,000-residue window offsets use an
  unseeded `np.random.randint`, so window positions are **not** reproducible on a rebuild.
- **Bulk-load PRAGMAs disable crash safety** (`journal_mode=OFF`, `synchronous=OFF`). An
  interrupted load leaves a possibly corrupt file — rebuild from scratch rather than resume.
