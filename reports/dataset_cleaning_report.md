# Dataset Cleaning Report

- Source metadata: `data/Analyse Critaria/metadata.json`
- Papers directory: `data/Research Papers`

## Counts

- Metadata papers: 7,253
- Missing markdown files: 1,385
- Papers with no real reviewer review: 0
- Cleaned papers kept: 5,868

## Derived Labels

- Good paper: 1,468
- Needs modification: 3,778
- Reject-risk: 622

## Cleaning Rules

- Removed author-only rows.
- Kept only reviews with numeric reviewer rating and useful text.
- Masked emails, URLs, OpenReview links, and reviewer IDs.
- Removed review blocks from paper text to preserve manuscript-only input.
- Masked acknowledgement sections when detected.
