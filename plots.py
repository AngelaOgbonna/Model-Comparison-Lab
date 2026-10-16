"""Plotly figures. Each model keeps the same color everywhere (color follows the entity, not its rank)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .modeling import METRIC_LABELS, METRICS, ExperimentResult

MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",
    "Random Forest": "#eb6834",
    "Gradient Boosting": "#1baf7a",
}
NEUTRAL = "#8a8986"
SEQUENTIAL_BLUE = [
    [0.0, "#cde2fb"],
    [0.25, "#86b6ef"],
    [0.5, "#3987e5"],
    [0.75, "#1c5cab"],
    [1.0, "#0d366b"],
]
FONT = dict(size=13)


def _layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        # Title sits at the very top of the figure; the legend gets its own row underneath it.
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top", yref="container"),
        height=height,
        margin=dict(l=10, r=10, t=95 if len(fig.data) > 1 else 55, b=10),
        font=FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(font_size=12),
    )
    return fig


def missing_values_chart(columns: pd.DataFrame) -> go.Figure:
    data = columns[columns["missing"] > 0].sort_values("missing_%")
    fig = go.Figure(
        go.Bar(
            x=data["missing_%"],
            y=data["column"],
            orientation="h",
            marker=dict(color=MODEL_COLORS["Logistic Regression"], cornerradius=4),
            text=[f"{v:.1f}%" for v in data["missing_%"]],
            textposition="outside",
            customdata=data["missing"],
            hovertemplate="<b>%{y}</b><br>%{x:.1f}% missing (%{customdata} rows)<extra></extra>",
        )
    )
    fig.update_xaxes(title="% of rows missing", range=[0, max(100, data["missing_%"].max() * 1.15)], showgrid=True)
    fig.update_yaxes(showgrid=False)
    return _layout(fig, "Missing values by column", height=max(220, 32 * len(data) + 90))


def class_balance_chart(balance: pd.DataFrame, target: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=balance["class"],
            y=balance["count"],
            marker=dict(color=MODEL_COLORS["Logistic Regression"], cornerradius=4),
            text=[f"{s:.1f}%" for s in balance["share_%"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y} rows (%{text})<extra></extra>",
        )
    )
    fig.update_xaxes(title=target, type="category")
    fig.update_yaxes(title="Rows", range=[0, balance["count"].max() * 1.2])
    return _layout(fig, f"Class balance of “{target}”", height=320)


def numeric_distribution(series: pd.Series, name: str) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=series.dropna(),
            marker=dict(color=MODEL_COLORS["Logistic Regression"], line=dict(width=1, color="white")),
            hovertemplate="%{x}<br>%{y} rows<extra></extra>",
        )
    )
    fig.update_xaxes(title=name)
    fig.update_yaxes(title="Rows")
    return _layout(fig, f"Distribution of “{name}”", height=300)


def metrics_comparison(exp: ExperimentResult, source: str = "cv") -> go.Figure:
    """Grouped bars: one group per metric, one bar per model. CV bars carry ±1 std error bars."""
    fig = go.Figure()
    labels = [METRIC_LABELS[m] for m in METRICS]
    for r in exp.models.values():
        if source == "cv":
            values = [r.cv_mean[m] for m in METRICS]
            err = dict(type="data", array=[r.cv_std[m] for m in METRICS], thickness=1.5, width=4, color=NEUTRAL)
        else:
            values = [r.test_metrics[m] for m in METRICS]
            err = None
        fig.add_bar(
            name=r.name,
            x=labels,
            y=values,
            error_y=err,
            marker=dict(color=MODEL_COLORS[r.name], cornerradius=4),
            hovertemplate=f"<b>{r.name}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
        )
    lowest = min(min(t.y) for t in fig.data)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(range=[max(0, lowest - 0.1), 1.0], title="Score")
    title = f"Cross-validated scores (mean ± 1 std, {exp.cv_folds} folds)" if source == "cv" else "Hold-out test scores"
    return _layout(fig, title)


def cv_folds_chart(exp: ExperimentResult, metric: str) -> go.Figure:
    """Fold-by-fold scores per model — shows the spread behind each mean."""
    fig = go.Figure()
    for r in exp.models.values():
        scores = r.cv_fold_scores[metric]
        fig.add_trace(
            go.Box(
                y=scores,
                name=r.name,
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                marker=dict(color=MODEL_COLORS[r.name], size=9, line=dict(width=2, color="white")),
                line=dict(color=MODEL_COLORS[r.name], width=2),
                fillcolor="rgba(0,0,0,0)",
                customdata=[f"Fold {i + 1}" for i in range(len(scores))],
                hovertemplate="<b>%{x}</b><br>%{customdata}: %{y:.3f}<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_yaxes(title=METRIC_LABELS[metric])
    return _layout(fig, f"{METRIC_LABELS[metric]} on each cross-validation fold", height=360)


def roc_chart(exp: ExperimentResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=NEUTRAL, dash="dash", width=1.5),
            name="Chance (AUC 0.5)",
            hoverinfo="skip",
        )
    )
    for r in exp.models.values():
        fpr, tpr = r.roc
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                line=dict(color=MODEL_COLORS[r.name], width=2),
                name=f"{r.name} (AUC {r.test_metrics['roc_auc']:.3f})",
                hovertemplate=f"<b>{r.name}</b><br>FPR %{{x:.3f}}<br>TPR %{{y:.3f}}<extra></extra>",
            )
        )
    suffix = "" if exp.is_binary else " — micro-averaged one-vs-rest"
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    fig = _layout(fig, f"ROC curves on hold-out set{suffix}", height=500)
    fig.update_layout(legend=dict(orientation="v", yanchor="bottom", y=0.03, xanchor="right", x=0.98))
    return fig


def confusion_chart(cm: np.ndarray, class_names: list[str], model: str, normalize: bool) -> go.Figure:
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    z = shares if normalize else cm
    text = [
        [f"{shares[i, j]:.0%}<br>({cm[i, j]})" if normalize else f"{cm[i, j]}<br>({shares[i, j]:.0%})"
         for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"{c}" for c in class_names],
            y=[f"{c}" for c in class_names],
            colorscale=SEQUENTIAL_BLUE,
            zmin=0,
            zmax=1 if normalize else None,
            text=text,
            texttemplate="%{text}",
            xgap=2,
            ygap=2,
            showscale=False,
            customdata=cm,
            hovertemplate="Actual <b>%{y}</b> → predicted <b>%{x}</b><br>%{customdata} rows<extra></extra>",
        )
    )
    fig.update_xaxes(title="Predicted", type="category", side="bottom")
    fig.update_yaxes(title="Actual", type="category", autorange="reversed")
    side = max(300, 70 * len(class_names) + 140)
    return _layout(fig, model, height=side)


def importance_chart(imp: pd.DataFrame, metric: str, model: str, top_n: int = 15) -> go.Figure:
    data = imp.head(top_n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["importance"],
            y=data["feature"],
            orientation="h",
            error_x=dict(type="data", array=data["std"], thickness=1.5, width=3, color=NEUTRAL),
            marker=dict(color=MODEL_COLORS.get(model, NEUTRAL), cornerradius=4),
            hovertemplate="<b>%{y}</b><br>Drop in score when shuffled: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title=f"Mean drop in {METRIC_LABELS[metric]} when the feature is shuffled")
    fig.update_yaxes(showgrid=False)
    return _layout(fig, f"Permutation importance — {model}", height=max(260, 28 * len(data) + 110))
