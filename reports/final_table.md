| model | seeds | trainable | minutes | val macro-F1 | test macro-F1 | test accuracy |
|---|---|---|---|---|---|---|
| lazy floor (always Governing Laws) | 1 | 0 | 0.0 | 0.0010 | 0.0011 | 0.0582 |
| TF-IDF + logistic regression | 1 | 17,347,800 | 2.3 | 0.8265 | 0.8291 | 0.8771 |
| BiLSTM + GloVe | 3 | 1,344,684 | 32.6 | 0.7895 ± 0.0044 | 0.7802 ± 0.0055 | 0.8486 |
| DistilBERT, full fine-tuning | 3 | 67,030,372 | 3.6 | 0.7908 ± 0.0002 | 0.7836 ± 0.0047 | 0.8673 |
| DistilBERT + LoRA r=8 (q, v) | 3 | 814,948 | 3.7 | 0.7841 ± 0.0008 | 0.7816 ± 0.0012 | 0.8584 |
| DistilBERT + LoRA r=16 (q, v) | 1 | 962,404 | 3.6 | 0.7848 | 0.7845 | 0.8580 |
| DistilBERT + LoRA r=8 (all linear) | 1 | 1,331,044 | 5.6 | 0.7905 | 0.7867 | 0.8605 |
| DistilBERT, full fine-tuning, 10 epochs | 3 | 67,030,372 | 11.6 | 0.8198 ± 0.0012 | 0.8270 ± 0.0036 | 0.8798 |
| DistilBERT + LoRA r=8 (q, v), 10 epochs | 3 | 814,948 | 11.9 | 0.8078 ± 0.0007 | 0.8144 ± 0.0005 | 0.8749 |
