import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# 1. Load the dataset
train_path = "data/drugLibTrain_raw.tsv"
test_path = "data/drugLibTest_raw.tsv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError(
        "Download both TSV files from the UCI dataset page "
        "and place them in the data/ folder."
    )

train_df = pd.read_csv(train_path, sep="\t")
test_df = pd.read_csv(test_path, sep="\t")
df = pd.concat([train_df, test_df], ignore_index=True)

# 2. Check required columns
required = ["sideEffectsReview", "sideEffects"]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns: {missing}. Found: {df.columns.tolist()}"
    )

# 3. Clean text and labels
df = df[required].copy()
df["sideEffectsReview"] = (
    df["sideEffectsReview"].fillna("").astype(str).str.strip()
)
df["sideEffects"] = (
    df["sideEffects"].fillna("").astype(str).str.strip()
)

# Remove records with missing text or target labels
df = df[
    (df["sideEffectsReview"] != "") &
    (df["sideEffects"] != "")
].copy()

# Remove accidental duplicate records
df = df.drop_duplicates()

# Normalize label whitespace
df["sideEffects"] = df["sideEffects"].str.replace(
    r"\s+", " ", regex=True
)

X = df["sideEffectsReview"]
y = df["sideEffects"]

print("Dataset shape:", df.shape)
print("\nTarget class distribution:")
print(y.value_counts())

# 4. Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Define models with TF-IDF text features (stop_words=None to preserve negation words like 'no' and 'not')
models = {
    "Multinomial Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 2),
            max_features=15000,
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB(alpha=0.5))
    ]),
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 2),
            max_features=15000,
            min_df=2,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

# 6. Train and evaluate
results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    results[name] = {
        "Accuracy": accuracy,
        "Macro F1": macro_f1
    }
    trained_models[name] = model
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

# 7. Compare models
results_df = pd.DataFrame(results).T
print("\nModel comparison:")
print(results_df.round(4))

# Select using macro F1, which gives each class equal weight
best_name = results_df["Macro F1"].idxmax()
best_model = trained_models[best_name]
print("\nBest model:", best_name)

# 8. Confusion matrix for the selected model
best_predictions = best_model.predict(X_test)
labels = sorted(y.unique())
cm = confusion_matrix(
    y_test, best_predictions, labels=labels
)

plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title(f"Confusion Matrix: {best_name}")
plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# 9. Save the full pipeline
joblib.dump(best_model, "side_effect_model.joblib")
print("\nSaved model to side_effect_model.joblib")
