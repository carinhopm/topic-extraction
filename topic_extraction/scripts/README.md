# Scripts

### Predict topics `predict.py`

This script allows predicting keyphrases using the novel topic extraction model.

```
python topic_extraction/scripts/predict.py -h
usage: predict.py [-h] [--lang LANG] --sources SOURCES [SOURCES ...]
                  [--output_files OUTPUT_FILES [OUTPUT_FILES ...]]
                  [--ke_method KE_METHOD] [--keyword_len KEYWORD_LEN]
                  [--num_keywords NUM_KEYWORDS] [--input_col INPUT_COL]
                  [--keep_text] [--ke_score] [--f_sem F_SEM]

Predict with Topic Extraction model and save the results in one of the
supported formats.

optional arguments:
  -h, --help            show this help message and exit
  --lang LANG           language (should be DA, SV, NO or EN)
  --sources SOURCES [SOURCES ...]
                        path to source files containing texts (should be
                        *.csv)
  --output_files OUTPUT_FILES [OUTPUT_FILES ...]
                        path to output files (should be *.csv)
  --ke_method KE_METHOD
                        KE method to use for prediction
  --keyword_len KEYWORD_LEN
                        max. number of words per keyphrase
  --num_keywords NUM_KEYWORDS
                        number of keywords per article
  --input_col INPUT_COL
                        which column should be considered as input
  --keep_text           Keep article body into results
  --ke_score            Return KE score per keyword
  --f_sem F_SEM         Semantic factor value
```

### Evaluate topic predictions `evaluate.py`

This script allows evaluating predicted keyphrases.

```
python topic_extraction/scripts/evaluate.py -h
usage: evaluate.py [-h] [--lang LANG] --sources SOURCES [SOURCES ...]
                   [--output_files OUTPUT_FILES [OUTPUT_FILES ...]]
                   [--pred_col PRED_COL] [--keywords_col KEYWORDS_COL]
                   [--all_preds] [--metrics METRICS [METRICS ...]]
                   [--num_preds NUM_PREDS]

Evaluate some Keyword Extraction method

optional arguments:
  -h, --help            show this help message and exit
  --lang LANG           language (should be DA, SV, NO or EN)
  --sources SOURCES [SOURCES ...]
                        paths to source files containing the predictions to
                        evaluate (should be *.csv)
  --output_files OUTPUT_FILES [OUTPUT_FILES ...]
                        path to output files (should be *.csv)
  --pred_col PRED_COL   which column contains the predictions to evaluate
  --keywords_col KEYWORDS_COL
                        which column contains the human-annotated keywords
  --all_preds           evaluate all available predictions from the CSV
  --metrics METRICS [METRICS ...]
                        which metrics to use for evaluation (Precision, F1,
                        1st_correct_pred, avg_precision, nDCG)
  --num_preds NUM_PREDS
                        number of predictions to evaluate per document
```
