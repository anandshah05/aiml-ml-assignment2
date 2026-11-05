"""Streamlit web app for early stage diabetes risk classification."""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DEFAULT_TEST_PATH = os.path.join(BASE_DIR, 'test_data.csv')
FULL_DATA_PATH = os.path.join(BASE_DIR, 'diabetes_full_data.csv')


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

MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Support Vector Machine': 'svm.joblib',
}

MODEL_OBSERVATIONS = {
    'Logistic Regression': (
        'Delivers strong results with `94.23%` accuracy and high precision.\n'
        'Identifies clear symptoms well, but misses a few cases with complex'
        ' symptom combinations.'
    ),
    'Decision Tree': (
        'Best performing model with `99.04%` accuracy and zero false'
        ' positives.\nSimple if-else decision rules match the yes/no symptom'
        ' questions perfectly.'
    ),
    'KNN': (
        'Achieves `93.27%` accuracy, ranking lowest among the five'
        ' models.\nTreating all symptoms equally allows less important symptoms'
        ' to reduce accuracy.'
    ),
    'Naive Bayes': (
        'Delivers fast and reliable results with `94.23%` accuracy and good'
        ' recall.\nAssuming symptoms are independent leads to a few false'
        ' alarms when symptoms overlap.'
    ),
    'Support Vector Machine': (
        'Outstanding performance with `98.08%` accuracy and the highest AUC'
        ' score of 0.9973.\nAccurately separates positive and negative cases'
        ' even with complex symptom patterns.'
    ),
}


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


def build_classifiers() -> dict:
  """Initializes the five classification models."""
  return {
      'Logistic Regression': LogisticRegression(
          C=1.0, max_iter=1000, random_state=42
      ),
      'Decision Tree': DecisionTreeClassifier(
          criterion='gini', random_state=42
      ),
      'KNN': KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
      'Naive Bayes': GaussianNB(),
      'Support Vector Machine': SVC(
          kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42
      ),
  }


def train_and_save_models() -> tuple:
  """Trains and saves models and scaler."""
  os.makedirs(MODEL_DIR, exist_ok=True)
  df_full = pd.read_csv(FULL_DATA_PATH)
  y_all = encode_target(df_full[TARGET_COLUMN])

  train_df, test_df = train_test_split(
      df_full, test_size=0.20, random_state=42, stratify=y_all
  )
  test_df.to_csv(DEFAULT_TEST_PATH, index=False)

  x_train = preprocess_features(train_df)
  y_train = encode_target(train_df[TARGET_COLUMN])

  scaler = StandardScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

  models = build_classifiers()
  for name, model in models.items():
    model.fit(x_train_scaled, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, MODEL_FILES[name]))

  return scaler, models


@st.cache_resource
def load_artifacts():
  """Loads saved models and scaler from disk."""
  scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
  if not os.path.exists(scaler_path) or not os.path.exists(DEFAULT_TEST_PATH):
    return train_and_save_models()

  scaler = joblib.load(scaler_path)
  models = {}
  for name, filename in MODEL_FILES.items():
    models[name] = joblib.load(os.path.join(MODEL_DIR, filename))
  return scaler, models


def evaluate_model_on_data(
    model, x_scaled: np.ndarray, y_true: np.ndarray
) -> dict:
  """Computes evaluation metrics and ROC curves."""
  y_pred = model.predict(x_scaled)
  y_prob = model.predict_proba(x_scaled)[:, 1]
  fpr, tpr, _ = roc_curve(y_true, y_prob)
  return {
      'Accuracy': float(accuracy_score(y_true, y_pred)),
      'AUC': float(roc_auc_score(y_true, y_prob)),
      'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
      'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
      'F1': float(f1_score(y_true, y_pred, zero_division=0)),
      'MCC': float(matthews_corrcoef(y_true, y_pred)),
      'confusion_matrix': confusion_matrix(y_true, y_pred),
      'fpr': fpr,
      'tpr': tpr,
  }


def main():
  """Runs the Streamlit application."""
  # Page configuration
  st.set_page_config(
      page_title='ML Assignment 2 - Early Stage Diabetes Classification',
      page_icon='🩺',
      layout='wide',
  )

  # Sidebar: Test dataset upload and model selection
  st.sidebar.header('Settings')

  st.sidebar.subheader('Upload Test Dataset')
  uploaded_file = st.sidebar.file_uploader(
      'Upload Test Data (CSV)',
      type=['csv'],
      help='Upload test dataset (CSV) containing the 16 features and class label.',
  )

  # Load models and scaler
  scaler, models = load_artifacts()

  # Load test data from uploaded file or default
  if uploaded_file is not None:
    try:
      df_uploaded = pd.read_csv(uploaded_file)
      missing_cols = [c for c in FEATURE_COLUMNS if c not in df_uploaded.columns]
      if missing_cols:
        st.sidebar.error(
            f'Missing required features: {", ".join(missing_cols[:3])}...'
        )
        st.sidebar.info('Using default test data.')
        df_test = pd.read_csv(DEFAULT_TEST_PATH)
        data_source = 'Default Test Dataset'
      else:
        df_test = df_uploaded
        data_source = f'Uploaded: {uploaded_file.name}'
        st.sidebar.success(f'Uploaded: {len(df_test)} samples')
    except Exception as e:
      st.sidebar.error(f'Error reading CSV: {e}')
      df_test = pd.read_csv(DEFAULT_TEST_PATH)
      data_source = 'Default Test Dataset'
  else:
    df_test = pd.read_csv(DEFAULT_TEST_PATH)
    data_source = 'Default Test Dataset (104 samples)'
    st.sidebar.info('Using default test dataset.')

  # Model selection dropdown
  st.sidebar.subheader('Select Model')
  model_options = ['All Models'] + list(MODEL_FILES.keys())
  selected_model = st.sidebar.selectbox(
      'Choose Model to View',
      options=model_options,
      index=0,
  )

  st.title('ML Assignment 2 - Early Stage Diabetes Classification')
  st.markdown("""
Comparison of **5 Machine Learning Classification Models** (**Logistic Regression**, **Decision Tree**, **KNN**, **Naive Bayes**, and **Support Vector Machine**) across **6 Evaluation Metrics** (Accuracy, AUC, Precision, Recall, F1, and MCC).

### Dataset Description
- **Dataset Name**: [Early Stage Diabetes Risk Prediction Dataset (`diabetes_data_upload.csv`)](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset)
- **Dataset Dimensions**: **520 instances** and **17 attributes** (**16 input features** + **1 binary target variable** `class`)
- **Missing Values**: **0 missing values** across all 17 columns
- **Train / Test Split**: Stratified 80% / 20% split (`random_state=42`), yielding **416 training samples** and **104 held-out test samples** (`test_data.csv`)
- **Target Variable (`class`) Distribution**:
  - `Positive` (Diabetic Risk, encoded as `1`): **320 instances (61.54%)** — 256 in train, 64 in test
  - `Negative` (Healthy Control, encoded as `0`): **200 instances (38.46%)** — 160 in train, 40 in test
""")

  has_target = TARGET_COLUMN in df_test.columns
  x_test = preprocess_features(df_test)
  x_test_scaled = scaler.transform(x_test)

  # Evaluate models on test data
  if has_target:
    y_test = encode_target(df_test[TARGET_COLUMN])
    all_eval_results = {
        m_name: evaluate_model_on_data(m_obj, x_test_scaled, y_test)
        for m_name, m_obj in models.items()
    }
  else:
    all_eval_results = {}
    st.warning(
        "Uploaded test data does not contain the 'class' column. Displaying"
        " model predictions."
    )

  with st.container():
    # Dataset metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Dataset Samples', '520')
    c2.metric('Training Split (80%)', '416')
    c3.metric('Test Set Samples', str(len(df_test)))
    c4.metric('Selected Model View', selected_model)

    if has_target:
      # Metrics comparison table
      st.subheader('Comparison Table — All 5 Classification Models')
      comparison_rows = [
          {
              'Model Name': m_name,
              'Accuracy': round(res['Accuracy'], 4),
              'AUC': round(res['AUC'], 4),
              'Precision': round(res['Precision'], 4),
              'Recall': round(res['Recall'], 4),
              'F1': round(res['F1'], 4),
              'MCC': round(res['MCC'], 4),
          }
          for m_name, res in all_eval_results.items()
      ]
      comp_df = pd.DataFrame(comparison_rows)
      st.dataframe(
          comp_df.style.highlight_max(
              subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
          ).format({
              'Accuracy': '{:.4f}',
              'AUC': '{:.4f}',
              'Precision': '{:.4f}',
              'Recall': '{:.4f}',
              'F1': '{:.4f}',
              'MCC': '{:.4f}',
          }),
          use_container_width=True,
          hide_index=True,
      )

      if selected_model == 'All Models':
        # Evaluation metric plots and ROC curves
        col_chart, col_roc = st.columns(2)
        with col_chart:
          st.markdown('#### Evaluation Metrics by Classifier')
          fig_bar, ax_bar = plt.subplots(figsize=(7.5, 4.5))
          metrics_list = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
          x_idx = np.arange(len(comp_df))
          width = 0.13
          colors = [
              '#1D4ED8',
              '#0EA5E9',
              '#10B981',
              '#F59E0B',
              '#8B5CF6',
              '#EC4899',
          ]
          for i, m_col in enumerate(metrics_list):
            offset = (i - 2.5) * width
            ax_bar.bar(
                x_idx + offset,
                comp_df[m_col],
                width,
                label=m_col,
                color=colors[i],
            )
          ax_bar.set_xticks(x_idx)
          ax_bar.set_xticklabels(
              comp_df['Model Name'], rotation=15, ha='right'
          )
          ax_bar.set_ylim(0.80, 1.03)
          ax_bar.set_ylabel('Score')
          ax_bar.legend(ncol=3, fontsize=8.5)
          ax_bar.grid(axis='y', linestyle='--', alpha=0.4)
          plt.tight_layout()
          st.pyplot(fig_bar)
          plt.close(fig_bar)

        with col_roc:
          st.markdown('#### ROC Curves Across All 5 Models')
          fig_roc, ax_roc = plt.subplots(figsize=(7.5, 4.5))
          for m_name, res in all_eval_results.items():
            ax_roc.plot(
                res['fpr'],
                res['tpr'],
                lw=2,
                label=f"{m_name} (AUC = {res['AUC']:.4f})",
            )
          ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5000)')
          ax_roc.set_xlabel('False Positive Rate')
          ax_roc.set_ylabel('True Positive Rate')
          ax_roc.legend(loc='lower right', fontsize=8.5)
          ax_roc.grid(alpha=0.35)
          plt.tight_layout()
          st.pyplot(fig_roc)
          plt.close(fig_roc)

        # Confusion matrices
        st.subheader('Confusion Matrices')
        fig_cm, axes_cm = plt.subplots(1, 5, figsize=(16, 3.2))
        for idx, (m_name, res) in enumerate(all_eval_results.items()):
          sns.heatmap(
              res['confusion_matrix'],
              annot=True,
              fmt='d',
              cmap='Blues',
              cbar=False,
              xticklabels=['Neg (0)', 'Pos (1)'],
              yticklabels=['Neg (0)', 'Pos (1)'],
              annot_kws={'size': 12, 'weight': 'bold'},
              ax=axes_cm[idx],
          )
          axes_cm[idx].set_title(
              f"{m_name}\nAcc: {res['Accuracy']:.2%}",
              fontsize=9.5,
              fontweight='bold',
          )
          axes_cm[idx].set_xlabel('Predicted')
          axes_cm[idx].set_ylabel('Actual' if idx == 0 else '')
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        # Model performance observations
        st.subheader('Observations from Training')
        obs_rows = [
            {
                'Model Name': m_name,
                'Observation': obs_text,
            }
            for m_name, obs_text in MODEL_OBSERVATIONS.items()
        ]
        obs_rows.append({
            'Model Name': 'Overall, Winner for your dataset?',
            'Observation': (
                'Decision Tree is the overall winner with `99.04%` accuracy and'
                ' near-perfect predictions.\nSupport Vector Machine is a close'
                ' runner-up with the best probability ranking.'
            ),
        })
        st.table(pd.DataFrame(obs_rows))

      else:
        # Selected individual model evaluation
        st.subheader(f'Detailed Performance — {selected_model}')
        res = all_eval_results[selected_model]

        # Metric cards for selected model
        m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
        m_c1.metric('Accuracy', f"{res['Accuracy']:.4f}")
        m_c2.metric('AUC Score', f"{res['AUC']:.4f}")
        m_c3.metric('Precision', f"{res['Precision']:.4f}")
        m_c4.metric('Recall', f"{res['Recall']:.4f}")
        m_c5.metric('F1 Score', f"{res['F1']:.4f}")
        m_c6.metric('MCC Score', f"{res['MCC']:.4f}")

        # Confusion Matrix and ROC Curve for selected model
        col_single_cm, col_single_roc = st.columns(2)
        with col_single_cm:
          st.markdown(f'#### Confusion Matrix — {selected_model}')
          fig_scm, ax_scm = plt.subplots(figsize=(5, 3.8))
          sns.heatmap(
              res['confusion_matrix'],
              annot=True,
              fmt='d',
              cmap='Blues',
              cbar=False,
              xticklabels=['Negative (0)', 'Positive (1)'],
              yticklabels=['Negative (0)', 'Positive (1)'],
              annot_kws={'size': 14, 'weight': 'bold'},
              ax=ax_scm,
          )
          ax_scm.set_title(
              f"{selected_model}\nAccuracy: {res['Accuracy']:.2%}",
              fontsize=11,
              fontweight='bold',
          )
          ax_scm.set_xlabel('Predicted Label')
          ax_scm.set_ylabel('Actual Label')
          plt.tight_layout()
          st.pyplot(fig_scm)
          plt.close(fig_scm)

        with col_single_roc:
          st.markdown(f'#### ROC Curve — {selected_model}')
          fig_sroc, ax_sroc = plt.subplots(figsize=(5.5, 3.8))
          ax_sroc.plot(
              res['fpr'],
              res['tpr'],
              lw=2.5,
              color='#2563EB',
              label=f"{selected_model} (AUC = {res['AUC']:.4f})",
          )
          ax_sroc.plot(
              [0, 1], [0, 1], 'k--', lw=1, label='Random Guess (AUC = 0.5000)'
          )
          ax_sroc.set_xlabel('False Positive Rate')
          ax_sroc.set_ylabel('True Positive Rate')
          ax_sroc.legend(loc='lower right', fontsize=9.5)
          ax_sroc.grid(alpha=0.35)
          plt.tight_layout()
          st.pyplot(fig_sroc)
          plt.close(fig_sroc)

        # Observation for selected model
        st.subheader(f'Observation for {selected_model}')
        st.info(MODEL_OBSERVATIONS[selected_model])

    else:
      # Inference without target labels
      st.subheader(f'Model Predictions on Uploaded Test Data ({selected_model})')
      active_models = (
          models
          if selected_model == 'All Models'
          else {selected_model: models[selected_model]}
      )
      pred_df = df_test.copy()
      for m_name, m_obj in active_models.items():
        pred_df[f'Predicted_{m_name}'] = m_obj.predict(x_test_scaled)
        pred_df[f'Predicted_{m_name}'] = pred_df[f'Predicted_{m_name}'].map(
            {1: 'Positive', 0: 'Negative'}
        )
      st.dataframe(pred_df, use_container_width=True)


if __name__ == '__main__':
  main()
