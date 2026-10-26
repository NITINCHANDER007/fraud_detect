"""
evaluate.py - Rigorous evaluation and visualization module for imbalanced fraud detection.
Computes PR-AUC, ROC-AUC, Precision, Recall, F1, optimal decision threshold, and generates plots.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Modern visualization style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Computes production fraud evaluation metrics at a specific decision threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # Business impact calculations
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 6),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "false_positive_rate": round(fpr, 5),
        "false_negative_rate": round(fnr, 5),
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Finds the decision threshold that maximizes the F1-Score on the precision-recall curve.
    Returns: (optimal_threshold, best_f1)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid zero division
    denominator = precisions + recalls
    f1_scores = np.where(denominator > 0, 2 * (precisions * recalls) / denominator, 0)

    # Note: precision_recall_curve returns thresholds of len(precisions) - 1
    best_idx = np.argmax(f1_scores[:-1])
    optimal_th = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    logger.info(f"Optimal decision threshold: {optimal_th:.4f} (Max F1: {best_f1:.4f})")
    return optimal_th, best_f1


def plot_pr_curves(models_probs: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: Path):
    """
    Plots comparative Precision-Recall curves.
    models_probs = {model_name: (y_true, y_prob)}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2563EB", "#10B981", "#F59E0B", "#EF4444"]
    for i, (name, (y_true, y_prob)) in enumerate(models_probs.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        color = colors[i % len(colors)]
        ax.plot(recall, precision, label=f"{name} (PR-AUC = {ap:.4f})", color=color, linewidth=2.2)

    # Baseline no-skill line = fraction of positive class
    any_y_true = next(iter(models_probs.values()))[0]
    no_skill = np.mean(any_y_true)
    ax.axhline(y=no_skill, color="#94A3B8", linestyle="--", label=f"No Skill ({no_skill:.4f})")

    ax.set_title("Precision-Recall Curve Comparison (Imbalanced Fraud Detection)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall (Detection Rate)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"PR curve plot saved to {output_path}")


def plot_roc_curves(models_probs: Dict[str, Tuple[np.ndarray, np.ndarray]], output_path: Path):
    """
    Plots comparative ROC curves.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2563EB", "#10B981", "#F59E0B", "#EF4444"]

    for i, (name, (y_true, y_prob)) in enumerate(models_probs.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, label=f"{name} (ROC-AUC = {auc:.4f})", color=color, linewidth=2.2)

    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", label="Random Chance (0.50)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall / TPR)", fontsize=11)
    ax.set_xlim([-0.01, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"ROC curve plot saved to {output_path}")


def plot_confusion_matrices(cm_dict: Dict[str, Tuple[int, int, int, int]], output_path: Path):
    """
    Plots comparative confusion matrices side-by-side.
    cm_dict = {model_name: (tn, fp, fn, tp)}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(cm_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, (tn, fp, fn, tp)) in zip(axes, cm_dict.items()):
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Predicted Legit", "Predicted Fraud"],
            yticklabels=["Actual Legit", "Actual Fraud"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Confusion matrix plot saved to {output_path}")


def plot_feature_importance(feature_names: list, importances: np.ndarray, output_path: Path, top_n: int = 15):
    """
    Plots top-N feature importance for tree-based models.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(top_n), top_values[::-1], color="#2563EB", edgecolor="#1E40AF", alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel("Relative Feature Importance (Gain)", fontsize=11)
    ax.set_title(f"Top {top_n} Most Discriminative Features in Fraud Scoring", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Feature importance plot saved to {output_path}")
