# PeerRead Final Evaluation

## Dataset

- Total rows: 4,492
- Labels: {'reject': 2536, 'accept': 1956}

## Model Evaluation

- Dev: {'rows': 225, 'accuracy': 0.6356, 'accept_precision': 0.5814, 'accept_recall': 0.5208, 'accept_f1': 0.5495, 'confusion': {'reject_as_reject': 93, 'reject_as_accept': 36, 'accept_as_reject': 46, 'accept_as_accept': 50}}
- Test: {'rows': 239, 'accuracy': 0.6109, 'accept_precision': 0.5797, 'accept_recall': 0.3846, 'accept_f1': 0.4624, 'confusion': {'reject_as_reject': 106, 'accept_as_reject': 64, 'reject_as_accept': 29, 'accept_as_accept': 40}}

## Decision Counts

- Predicted: {'Reject': 1321, 'Modify': 2823, 'Accept': 348}
- Actual: {'Reject': 2536, 'Accept': 1956}

## Completed Parts

- supervised PeerRead accept/reject model
- XAI explanations
- paper-by-paper table
- dataset summary
- section summaries
- SFT dataset
- poster figures
- dashboard
