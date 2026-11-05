"""Trains and evaluates classification models on the diabetes risk dataset."""

import json
import os
import warnings

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_PATH = os.path.join(DATA_DIR, 'diabetes_data_upload.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

FEATURE_COLUMNS = [
    'Age',
    'Gender',
    'Polyuria',
    'Polydipsia',
    'sudden weight loss',
    'weakness',
    'Polyphagia',
    'Genital thrush',
    'visual blurring',
    'Itching',
    'Irritability',
    'delayed healing',
    'partial paresis',
    'muscle stiffness',
    'Alopecia',
    'Obesity',
]
TARGET_COLUMN = 'class'
BINARY_SYMPTOM_COLUMNS = [
    col for col in FEATURE_COLUMNS if col not in ('Age', 'Gender')
]


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
  """Encodes categorical features to binary values."""
  x_df = df[FEATURE_COLUMNS].copy()
  if not pd.api.types.is_numeric_dtype(x_df['Gender']):
    x_df['Gender'] = (
        x_df['Gender']
        .astype(str)
        .str.strip()
        .str.lower()
        .map({'male': 1, 'female': 0, '1': 1, '0': 0})
        .astype(int)
    )
  for col in BINARY_SYMPTOM_COLUMNS:
    if not pd.api.types.is_numeric_dtype(x_df[col]):
      x_df[col] = (
          x_df[col]
          .astype(str)
          .str.strip()
          .str.lower()
          .map({'yes': 1, 'no': 0, '1': 1, '0': 0})
          .astype(int)
      )
  x_df['Age'] = pd.to_numeric(x_df['Age'])
  return x_df


def encode_target(series: pd.Series) -> np.ndarray:
  """Encodes target labels into binary 0 and 1."""
  if not pd.api.types.is_numeric_dtype(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({'positive': 1, 'negative': 0, '1': 1, '0': 0})
        .astype(int)
        .to_numpy()
    )
  return series.astype(int).to_numpy()

def main():
  """Trains models, evaluates performance, and saves artifacts."""
  os.makedirs(MODEL_DIR, exist_ok=True)

  # 1. Load and preprocess dataset
  df = pd.read_csv(DATA_PATH)
  x_encoded = preprocess_features(df)
  y_all = encode_target(df[TARGET_COLUMN])

  # 2. Stratified 80/20 train-test split
  train_df, test_df = train_test_split(
      df,
      test_size=0.20,
      random_state=42,
      stratify=y_all,
  )
  test_df.to_csv(TEST_DATA_PATH, index=False)

  x_train = preprocess_features(train_df)
  y_train = encode_target(train_df[TARGET_COLUMN])
  x_test = preprocess_features(test_df)
  y_test = encode_target(test_df[TARGET_COLUMN])

  # 3. Standardize features
  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.transform(x_test)
  joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

  # 4. Define classification models
  model_specs = {
      'Logistic Regression': (
          'logistic_regression.joblib',
          LogisticRegression(C=1.0, max_iter=1000, random_state=42),
      ),
      'Decision Tree': (
          'decision_tree.joblib',
          DecisionTreeClassifier(criterion='gini', random_state=42),
      ),
      'KNN': (
          'knn.joblib',
          KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
      ),
      'Naive Bayes': (
          'naive_bayes.joblib',
          GaussianNB(),
      ),
      'Support Vector Machine': (
          'svm.joblib',
          SVC(
              kernel='rbf',
              C=1.0,
              gamma='scale',
              probability=True,
              random_state=42,
          ),
      ),
  }

  results = {}
  roc_data = {}
  trained_models = {}
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  x_all_scaled = scaler.transform(x_encoded)

  # 5. Train and evaluate models
  for name, (filename, model) in model_specs.items():
    model.fit(x_train_scaled, y_train)
    trained_models[name] = model
    joblib.dump(model, os.path.join(MODEL_DIR, filename))

    y_pred = model.predict(x_test_scaled)
    y_prob = model.predict_proba(x_test_scaled)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    mcc = float(matthews_corrcoef(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

    # 5-fold cross-validation for stability analysis
    cv_scores = cross_validate(
        model,
        x_all_scaled,
        y_all,
        cv=cv,
        scoring=[
            'accuracy',
            'roc_auc',
            'precision',
            'recall',
            'f1',
            'matthews_corrcoef',
        ],
    )

    results[name] = {
        'filename': filename,
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4),
        'confusion_matrix': cm,
        'classification_report': classification_report(
            y_test,
            y_pred,
            target_names=['Negative (0)', 'Positive (1)'],
            output_dict=True,
        ),
        'cv_5fold_mean': {
            'Accuracy': round(float(np.mean(cv_scores['test_accuracy'])), 4),
            'AUC': round(float(np.mean(cv_scores['test_roc_auc'])), 4),
            'Precision': round(float(np.mean(cv_scores['test_precision'])), 4),
            'Recall': round(float(np.mean(cv_scores['test_recall'])), 4),
            'F1': round(float(np.mean(cv_scores['test_f1'])), 4),
            'MCC': round(
                float(np.mean(cv_scores['test_matthews_corrcoef'])), 4
            ),
        },
        'cv_5fold_std': {
            'Accuracy': round(float(np.std(cv_scores['test_accuracy'])), 4),
            'AUC': round(float(np.std(cv_scores['test_roc_auc'])), 4),
            'F1': round(float(np.std(cv_scores['test_f1'])), 4),
            'MCC': round(float(np.std(cv_scores['test_matthews_corrcoef'])), 4),
        },
    }

  # Baseline MultinomialNB comparison
  mnb = MultinomialNB()
  mnb.fit(x_train, y_train)
  mnb_pred = mnb.predict(x_test)
  mnb_prob = mnb.predict_proba(x_test)[:, 1]
  mnb_summary = {
      'Accuracy': round(float(accuracy_score(y_test, mnb_pred)), 4),
      'AUC': round(float(roc_auc_score(y_test, mnb_prob)), 4),
      'Precision': round(float(precision_score(y_test, mnb_pred)), 4),
      'Recall': round(float(recall_score(y_test, mnb_pred)), 4),
      'F1': round(float(f1_score(y_test, mnb_pred)), 4),
      'MCC': round(float(matthews_corrcoef(y_test, mnb_pred)), 4),
      'confusion_matrix': confusion_matrix(y_test, mnb_pred).tolist(),
  }

  # Save metadata and metrics summary
  metadata = {
      'dataset_name': 'Early Stage Diabetes Risk Prediction Dataset (UCI)',
      'total_samples': int(len(df)),
      'train_samples': int(len(train_df)),
      'test_samples': int(len(test_df)),
      'feature_columns': FEATURE_COLUMNS,
      'target_column': TARGET_COLUMN,
      'class_distribution': {
          'Positive': int((y_all == 1).sum()),
          'Negative': int((y_all == 0).sum()),
      },
      'models': results,
      'multinomial_nb_comparison': mnb_summary,
  }

  with open(os.path.join(MODEL_DIR, 'metrics_summary.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

  # Print summary table
  print('Training & evaluation completed successfully!')
  print(
      f"{'ML Model Name':<25} | {'Acc':<6} | {'AUC':<6} | {'Prec':<6} |"
      f" {'Rec':<6} | {'F1':<6} | {'MCC':<6}"
  )
  print('-' * 76)
  for name, m in results.items():
    print(
        f"{name:<25} | {m['Accuracy']:.4f} | {m['AUC']:.4f} |"
        f" {m['Precision']:.4f} | {m['Recall']:.4f} | {m['F1']:.4f} |"
        f" {m['MCC']:.4f}"
    )


if __name__ == '__main__':
  main()
