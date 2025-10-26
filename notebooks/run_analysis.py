"""
Run model analysis for ISL project and save outputs to analysis_outputs/
Run: python notebooks/run_analysis.py
"""
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_recall_fscore_support, balanced_accuracy_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# determine project root (one level up from notebooks/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set(style="whitegrid")


def savefig(fname, dpi=140):
    p = OUTPUT_DIR / fname
    plt.tight_layout()
    plt.savefig(p, dpi=dpi)
    print("Saved:", p)


# Load model
from tensorflow.keras.models import load_model
MODEL_PATH = PROJECT_ROOT / "model.h5"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"model.h5 not found at {MODEL_PATH}. Place your trained model as model.h5 in the project root ({PROJECT_ROOT}).")

model = load_model(MODEL_PATH)
print("Loaded model:", MODEL_PATH)

# Load test data
def load_test_data():
    # look for files in project root
    x_path = PROJECT_ROOT / "X_test.npy"
    y_path = PROJECT_ROOT / "y_test.npy"
    csv_path = PROJECT_ROOT / "keypoint.csv"

    if x_path.exists() and y_path.exists():
        X = np.load(x_path)
        y = np.load(y_path)
        print("Loaded X_test.npy and y_test.npy shapes:", X.shape, y.shape)
        return X, y

    if csv_path.exists():
        # Try reading with header. If no 'label' column, assume first column is label and file has no header.
        df = pd.read_csv(csv_path, header=0)
        if 'label' in df.columns:
            y = df['label'].values
            X = df.drop(columns=['label']).values
            print("Loaded keypoint.csv (with header) shapes:", X.shape, y.shape)
            return X, y

        # Fallback: read without header, treat first column as label
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] < 2:
            raise ValueError("keypoint.csv found but doesn't contain label + features columns.")
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
        print("Loaded keypoint.csv (no header) shapes:", X.shape, y.shape)
        return X, y

    raise FileNotFoundError("No test data found. Place X_test.npy & y_test.npy or keypoint.csv in project root.")

X_test, y_test = load_test_data()
print("Raw X_test shape:", X_test.shape)

# Encode/prepare labels
if y_test.dtype.kind in {'U','S','O'}:
    le = LabelEncoder().fit(y_test)
    y_encoded = le.transform(y_test)
    classes = list(le.classes_)
    print("Text labels found. Classes:", classes)
else:
    y_encoded = y_test.astype(int)
    classes = [str(i) for i in sorted(np.unique(y_encoded))]
    print("Integer labels. Classes:", classes)

n_classes = len(classes)
print("n_classes (from y):", n_classes)

# Reshape X if necessary
input_shape = model.input_shape
print("Model input shape:", input_shape)
if len(input_shape) == 3 and X_test.ndim == 2:
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    print("Reshaped X_test to:", X_test.shape)

# Predict
print("Running predictions...")
y_prob = model.predict(X_test, batch_size=32)
if y_prob.ndim == 1 or (y_prob.ndim == 2 and y_prob.shape[1] == 1):
    y_prob = np.vstack([1 - y_prob, y_prob]).T
y_pred = np.argmax(y_prob, axis=1)
print("Predictions done. y_pred shape:", y_pred.shape)

# If model predicts more classes than appear in y_test, extend class names
pred_n_classes = y_prob.shape[1]
if pred_n_classes != n_classes:
    print(f"Warning: model predicts {pred_n_classes} classes but y contains {n_classes} classes.\n" \
          "Extending class names for reporting (missing classes will show zero support).")
    # extend classes list with placeholders
    for i in range(n_classes, pred_n_classes):
        classes.append(f"class_{i}")
    n_classes = pred_n_classes
    print("Adjusted classes list length to", n_classes)

# Confusion matrix
cm = confusion_matrix(y_encoded, y_pred, labels=np.arange(n_classes))
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (counts)")
plt.ylabel("True")
plt.xlabel("Predicted")
savefig("confusion_matrix_counts.png")
plt.close()

# Normalized cm
row_sums = cm.sum(axis=1)
safe_div = row_sums.copy().astype(float)
safe_div[safe_div == 0] = 1.0
cm_norm = cm.astype('float') / safe_div[:, np.newaxis]
cmn_df = pd.DataFrame(np.round(cm_norm, 3), index=classes, columns=classes)
plt.figure(figsize=(10,8))
sns.heatmap(cmn_df, annot=True, fmt='.3f', cmap='rocket')
plt.title("Confusion Matrix (normalized)")
plt.ylabel("True")
plt.xlabel("Predicted")
savefig("confusion_matrix_normalized.png")
plt.close()

# Classification report
labels_for_report = np.arange(n_classes)
report = classification_report(y_encoded, y_pred, labels=labels_for_report, target_names=classes, output_dict=True, zero_division=0)
print("Classification report:\n", classification_report(y_encoded, y_pred, labels=labels_for_report, target_names=classes, zero_division=0))
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(OUTPUT_DIR / "classification_report.csv")
print("Saved classification_report.csv")

# summary stats
acc = accuracy_score(y_encoded, y_pred)
bal_acc = balanced_accuracy_score(y_encoded, y_pred)
summary = {
    "accuracy": float(acc),
    "balanced_accuracy": float(bal_acc)
}
with open(OUTPUT_DIR / "summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved summary_stats.json")

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(y_encoded, y_pred, labels=np.arange(n_classes))
metrics_df = pd.DataFrame({
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "support": support
}, index=classes)

plt.figure(figsize=(12,6))
metrics_df[['precision','recall','f1']].plot(kind='bar')
plt.title("Per-class precision / recall / f1")
savefig("per_class_prf.png")
plt.close()

plt.figure(figsize=(10,4))
metrics_df['support'].sort_values(ascending=False).plot(kind='bar')
plt.title("Class support")
savefig("class_support.png")
plt.close()
metrics_df.to_csv(OUTPUT_DIR / "per_class_metrics.csv")
print("Saved per_class_metrics.csv")

# ROC & PR
if n_classes > 2:
    y_bin = label_binarize(y_encoded, classes=np.arange(n_classes))
else:
    y_bin = np.vstack([1 - y_encoded, y_encoded]).T

plt.figure(figsize=(10,8))
for i in range(n_classes):
    if y_prob.shape[1] <= i:
        continue
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{classes[i]} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (one-vs-rest)")
plt.legend(loc='lower right', fontsize='small')
savefig("roc_curves.png")
plt.close()

plt.figure(figsize=(10,8))
for i in range(n_classes):
    precision_vals, recall_vals, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
    ap = auc(recall_vals, precision_vals)
    plt.plot(recall_vals, precision_vals, lw=2, label=f"{classes[i]} (AP={ap:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend(loc='upper right', fontsize='small')
savefig("pr_curves.png")
plt.close()

# Embeddings
from tensorflow.keras import Model

def get_penultimate_embeddings(model, X):
    try:
        penult_output = model.layers[-2].output
        embed_model = Model(inputs=model.input, outputs=penult_output)
        embeddings = embed_model.predict(X, batch_size=64)
        return embeddings
    except Exception as e:
        print("Embedding extraction failed, falling back to probabilities:", e)
        return y_prob

embeddings = get_penultimate_embeddings(model, X_test)
print("Embeddings shape:", embeddings.shape)
emb_flat = embeddings.reshape((embeddings.shape[0], -1))

pca = PCA(n_components=2)
emb_pca = pca.fit_transform(emb_flat)
plt.figure(figsize=(8,6))
for i, cls in enumerate(classes):
    idx = np.where(y_encoded == i)
    plt.scatter(emb_pca[idx,0], emb_pca[idx,1], label=cls, alpha=0.6, s=20)
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
plt.title("PCA of embeddings")
savefig("embeddings_pca.png")
plt.close()

# t-SNE (subset if large)
from sklearn.manifold import TSNE
subset = emb_flat
if emb_flat.shape[0] > 2000:
    subset = emb_flat[:2000]
    lab_subset = y_encoded[:2000]
else:
    lab_subset = y_encoded

tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto', max_iter=1000)
emb_tsne = tsne.fit_transform(subset)
plt.figure(figsize=(8,6))
for i, cls in enumerate(classes):
    idx = np.where(lab_subset == i)
    plt.scatter(emb_tsne[idx,0], emb_tsne[idx,1], label=cls, alpha=0.6, s=15)
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
plt.title("t-SNE of embeddings")
savefig("embeddings_tsne.png")
plt.close()

print("All done. Outputs saved to:", OUTPUT_DIR.resolve())
