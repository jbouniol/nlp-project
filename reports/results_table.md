| run | model | seed | trainable_params | train_time_s | val_macro_f1 | test_macro_f1 | test_accuracy |
|---|---|---|---|---|---|---|---|
| lazy | lazy | 0 | 0 | 0 | 0.0010 | 0.0011 | 0.0582 |
| lstm_seed0 | lstm | 0 | 1,344,684 | 2,055 | 0.7900 | 0.7799 | 0.8469 |
| lstm_seed1 | lstm | 1 | 1,344,684 | 1,721 | 0.7936 | 0.7859 | 0.8512 |
| lstm_seed2 | lstm | 2 | 1,344,684 | 2,097 | 0.7848 | 0.7749 | 0.8477 |
| tfidf | tfidf | 0 | 17,347,800 | 139 | 0.8265 | 0.8291 | 0.8771 |
