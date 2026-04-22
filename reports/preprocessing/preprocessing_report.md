# Preprocessing Report

## Raw Files
- Jigsaw: `/Users/zhengpeixian/ZPX/UTS/NLP/Assignment3/toxic_comment_detection/dataset/jigsaw/raw/train.csv`
- Civil Comments: `missing_optional`

## Schema Mapping Decisions
- jigsaw: text=`comment_text`, labels=`['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']`
- civil_comments: text=`None`, labels=`None`

## Cleaning Rules
- Unicode NFKC normalization, HTML entity cleanup, line-break replacement, whitespace collapse, and stripping.
- URLs and emails are replaced with stable placeholders.
- No stemming, lemmatization, stopword removal, profanity masking, spelling correction, or aggressive punctuation stripping.

## Duplicate Removal Summary
- jigsaw: input=159571, removed_missing_text=0, removed_missing_labels=0, removed_normalized_duplicates=258, output=159313

## Split Sizes
- jigsaw: method=two-stage StratifiedShuffleSplit, stratify_column=binary_label, counts=test: 23897, train: 111519, val: 23897

## Teammate-Facing Exports
- `data/processed/train.csv`: 111519 rows, columns=`raw_text`, `clean_text`, `label`
- `data/processed/val.csv`: 23897 rows, columns=`raw_text`, `clean_text`, `label`
- `data/processed/test.csv`: 23897 rows, columns=`raw_text`, `clean_text`, `label`

## Label Distributions
- jigsaw: {'row_count': 159313, 'label_counts': {'0': 143138, '1': 16175}, 'positive_ratio': 0.10152969311983329}

## Token Length Statistics
- jigsaw: {'available': True, 'tokenizer_name': 'distilbert-base-uncased', 'mean': 93.64426004155342, 'median': 51.0, 'p90': 204.0, 'p95': 307.0, 'fraction_gt_128': 0.18985895689617294, 'fraction_gt_256': 0.0678977861191491}

## Leakage Checks
- jigsaw raw overlap: {'test__train': {'overlap_count': 0, 'example_hashes': []}, 'test__val': {'overlap_count': 0, 'example_hashes': []}, 'train__val': {'overlap_count': 0, 'example_hashes': []}}
- jigsaw normalized overlap: {'test__train': {'overlap_count': 0, 'example_hashes': []}, 'test__val': {'overlap_count': 0, 'example_hashes': []}, 'train__val': {'overlap_count': 0, 'example_hashes': []}}

## Slice Summaries
- jigsaw: {'has_identity_term': {'0': {'row_count': 149967, 'positive_ratio': 0.09580107623677209, 'label_counts': {'0': 135600, '1': 14367}}, '1': {'row_count': 9346, 'positive_ratio': 0.19345174406163065, 'label_counts': {'0': 7538, '1': 1808}}}, 'has_obfuscation': {'0': {'row_count': 92860, 'positive_ratio': 0.07472539306482877, 'label_counts': {'0': 85921, '1': 6939}}, '1': {'row_count': 66453, 'positive_ratio': 0.13898544836200022, 'label_counts': {'0': 57217, '1': 9236}}}, 'implicit_proxy': {'0': {'row_count': 149102, 'positive_ratio': 0.03999946345454789, 'label_counts': {'0': 143138, '1': 5964}}, '1': {'row_count': 10211, 'positive_ratio': 1.0, 'label_counts': {'1': 10211}}}, 'length_bucket': {'long': {'row_count': 53929, 'positive_ratio': 0.06582729143874354, 'label_counts': {'0': 50379, '1': 3550}}, 'medium': {'row_count': 51590, 'positive_ratio': 0.0906183368869936, 'label_counts': {'0': 46915, '1': 4675}}, 'short': {'row_count': 53794, 'positive_ratio': 0.14778599843848755, 'label_counts': {'0': 45844, '1': 7950}}}}

## Assumptions
- Jigsaw is the only dataset for the main non-extension experiment.
- Civil Comments artifacts are optional extension outputs and may be dropped without breaking the Jigsaw pipeline.
- Jigsaw `identity_hate` is treated as an original toxicity label, not as a dataset-provided identity mention column.
- Length buckets use empirical Jigsaw word-length quantiles and are then reused for Civil Comments for comparability.

## Warnings
- Civil Comments file missing; optional extension artifacts skipped.
