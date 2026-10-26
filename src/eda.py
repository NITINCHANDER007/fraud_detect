"""
eda.py - Exploratory Data Analysis module for Credit Card Fraud Detection.
Generates publication-quality figures showcasing extreme class imbalance,
transaction amount distributions, and feature correlations with the fraud target.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_dataset

REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def run_eda():
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")

    # 1. Class Distribution Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df["Class"].value_counts()
    percentages = df["Class"].value_counts(normalize=True) * 100

    colors = ["#2563EB", "#DC2626"]
    axes[0].bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Count (Log Scale)", fontsize=11)
    axes[0].set_title("Transaction Count by Class (Log Scale)", fontsize=12, fontweight="bold")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v * 1.3, f"{v:,}\n({percentages.iloc[i]:.3f}%)", ha="center", fontweight="bold")

    # Ratio representation
    legit_ratio = counts[0] / counts[1]
    axes[1].pie(
        [counts[0], counts[1]],
        labels=["Legitimate (99.83%)", "Fraud (0.17%)"],
        colors=colors,
        explode=(0, 0.25),
        autopct="%1.2f%%",
        startangle=40,
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    axes[1].set_title(f"Class Imbalance: 1 Fraud per {legit_ratio:.0f} Transactions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "class_distribution.png", dpi=300)
    plt.close(fig)
    print("Saved class_distribution.png")

    # 2. Transaction Amount Distribution (Legit vs Fraud)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    legit_amt = df[df["Class"] == 0]["Amount"]
    fraud_amt = df[df["Class"] == 1]["Amount"]

    sns.histplot(
        np.log1p(legit_amt),
        ax=axes[0],
        color="#2563EB",
        kde=True,
        stat="density",
        label=f"Legit (Median: ${legit_amt.median():.2f})",
        alpha=0.4,
    )
    sns.histplot(
        np.log1p(fraud_amt),
        ax=axes[0],
        color="#DC2626",
        kde=True,
        stat="density",
        label=f"Fraud (Median: ${fraud_amt.median():.2f})",
        alpha=0.6,
    )
    axes[0].set_title("Log(1 + Amount) Distribution Comparison", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Log(1 + Transaction Amount in USD)", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].legend()

    # Boxplot with outliers visible
    df_plot = df.copy()
    df_plot["Class_Label"] = df_plot["Class"].map({0: "Legitimate", 1: "Fraud"})
    sns.boxplot(
        x="Class_Label",
        y="Amount",
        data=df_plot,
        ax=axes[1],
        palette={"Legitimate": "#2563EB", "Fraud": "#DC2626"},
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Transaction Amount Boxplot (Log Scale)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Amount (USD, Log Scale)", fontsize=11)
    axes[1].set_xlabel("Class", fontsize=11)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "amount_distribution.png", dpi=300)
    plt.close(fig)
    print("Saved amount_distribution.png")

    # 3. Correlation Heatmap of Top Discriminative PCA Features
    corrs = df.corr()["Class"].drop("Class")
    top_pos = corrs.sort_values(ascending=False).head(6)
    top_neg = corrs.sort_values().head(6)
    top_corr_features = list(top_pos.index) + list(top_neg.index)

    fig, ax = plt.subplots(figsize=(10, 7))
    subset_corr = df[top_corr_features + ["Class"]].corr()
    sns.heatmap(
        subset_corr,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
    )
    ax.set_title("Correlation Heatmap: Top 12 Predictive Features with Fraud Class", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=300)
    plt.close(fig)
    print("Saved correlation_heatmap.png")


if __name__ == "__main__":
    run_eda()
