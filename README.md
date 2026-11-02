# ML Model Evaluation & Data-Quality Toolkit

Simple Python toolkit to evaluate classification models and run basic data quality checks.

**Status:** Completed (learning project)

## What it does

- Stratified train/test split
- Scaling fitted only on training data (avoids leakage)
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Data checks for non-finite values, class balance, and train/test overlap
- Pytest tests included

## Project structure

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

## Notes

Currently uses synthetic data for demonstration.  
Next improvement: add a real public dataset and compare a couple of models.

## Author

Vinay Adabala  
GitHub: [vinayadabala05](https://github.com/vinayadabala05)
