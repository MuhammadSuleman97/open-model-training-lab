# Dataset sources

## BANKING77

- Canonical source: `PolyAI-LDN/task-specific-datasets`
- Pinned revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- Task: classify English online-banking questions into 77 fine-grained intents
- Verified splits: 10,003 training examples and 3,080 test examples
- Source format: CSV with `text` and `category`, plus `categories.json`
- Licence: CC BY 4.0

The previously considered `mteb/banking77` mirror at revision
`18072d2685ea682290f7b8924d94c62acc19c0b2` was rejected after direct
inspection. Its Parquet files contain 9,993 training rows and 3,076 test rows,
silently omitting 14 unique queries from the original published splits. Its
dataset card still reports the original 10,003/3,080 statistics.

The lab therefore downloads the original files directly from the pinned PolyAI
repository commit and verifies their SHA-256 checksums. Keep the original
attribution and review licensing before redistributing data or trained
artifacts.
