"""
Evaluation utilities.

Reports the metrics an interviewer will actually ask about:
  - accuracy        (easy to explain, but misleading alone on imbalanced data)
  - precision/recall/F1 (class-level performance)
  - log-loss         (the *actual* metric the Kaggle competition was scored on)
  - confusion matrix
"""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix,
)


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "log_loss": round(log_loss(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def print_comparison_table(results: dict):
    print("\n" + "=" * 72)
    print(f"{'Model':<20}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}{'LogLoss':>10}")
    print("-" * 72)
    for name, m in results.items():
        print(f"{name:<20}{m['accuracy']:>10}{m['precision']:>12}"
              f"{m['recall']:>10}{m['f1_score']:>10}{m['log_loss']:>10}")
    print("=" * 72)
