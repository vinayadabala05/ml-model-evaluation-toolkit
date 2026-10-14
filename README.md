# ML Model Evaluation & Data-Quality Toolkit

A clean, practical Python toolkit for evaluating classification models and checking data quality.

## What this project does

- Implements a classification baseline with **stratified holdout evaluation**
- Fits preprocessing (scaling) **only on training data** to avoid data leakage
- Reports standard metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Performs data quality checks (non-finite values, class distribution, train/test overlap)
- Includes automated tests using **pytest**

## Project Structure

```
ml-model-evaluation-toolkit/
├── src/
│   ├── data_checks.py
│   ├── evaluation.py
│   └── pipeline.py
├── tests/
│   └── test_evaluation.py
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python -m src.evaluation
pytest tests/ -v
```

## Key Features

- Stratified train/test split
- Preprocessing pipeline that fits only on train set
- Full classification metrics report
- Data quality checks (NaN/Inf, class counts, index overlap)
- Automated unit tests for metric ranges and split integrity

## Author

**Vinay Adabala**  
GitHub: [vinayadabala05](https://github.com/vinayadabala05)
