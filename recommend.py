"""Turn experiment results into a recommendation with a plain-language rationale."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modeling import METRIC_LABELS, METRICS, MODEL_SPECS, ExperimentResult

# A train-vs-CV gap larger than this suggests the model is memorising the training folds.
OVERFIT_GAP = 0.05


@dataclass
class Recommendation:
    model: str
    top_scorer: str
    used_one_se_rule: bool
    reasons: list[str]
    cautions: list[str]

    def to_markdown(self) -> str:
        lines = [f"### Recommended model: **{self.model}**", ""]
        lines += [f"- {r}" for r in self.reasons]
        if self.cautions:
            lines += ["", "**Things to keep in mind**", ""]
            lines += [f"- {c}" for c in self.cautions]
        return "\n".join(lines)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else many or one + 's'}"


def _describe_params(params: dict) -> str:
    return ", ".join(f"`{k}={v}`" for k, v in params.items())


def recommend(exp: ExperimentResult, imbalance_ratio: float | None = None) -> Recommendation:
    metric = exp.primary_metric
    label = METRIC_LABELS[metric]
    k = exp.cv_folds
    ranked = sorted(exp.models.values(), key=lambda r: r.cv_mean[metric], reverse=True)
    top = ranked[0]

    # One-standard-error rule: any model whose CV mean is within one standard error of the best
    # is statistically indistinguishable from it, so prefer the simplest of those.
    se = top.cv_std[metric] / np.sqrt(k)
    threshold = top.cv_mean[metric] - se
    contenders = [r for r in ranked if r.cv_mean[metric] >= threshold]
    chosen = min(contenders, key=lambda r: (MODEL_SPECS[r.name].complexity, -r.cv_mean[metric]))
    used_one_se = chosen.name != top.name

    reasons: list[str] = []
    cautions: list[str] = []

    reasons.append(
        f"Selection criterion: mean **{label}** across {k} stratified cross-validation folds on the "
        f"training set ({exp.n_train} rows). Cross-validation is used rather than the single hold-out "
        "split because averaging over several folds gives a lower-variance estimate."
    )

    reasons.append(
        f"**{chosen.name}** scored a CV {label} of **{_fmt(chosen.cv_mean[metric])} ± "
        f"{_fmt(chosen.cv_std[metric])}** and **{_fmt(chosen.test_metrics[metric])}** on the untouched "
        f"hold-out set ({exp.n_test} rows)."
    )

    if used_one_se:
        reasons.append(
            f"**{top.name}** had the highest raw CV {label} ({_fmt(top.cv_mean[metric])}), but "
            f"{chosen.name} is within one standard error ({_fmt(se)}) of it. Under the one-standard-error "
            f"rule the simpler model is preferred: it is cheaper to run, easier to explain and less "
            "likely to overfit, for a difference the data cannot reliably resolve."
        )
    elif len(ranked) > 1:
        runner = ranked[1]
        margin = top.cv_mean[metric] - runner.cv_mean[metric]
        pairs = list(zip(top.cv_fold_scores[metric], runner.cv_fold_scores[metric], strict=True))
        wins = sum(a > b + 1e-9 for a, b in pairs)
        losses = sum(b > a + 1e-9 for a, b in pairs)
        ties = k - wins - losses
        record = f"{_plural(wins, 'win')}, {_plural(ties, 'tie')} and {_plural(losses, 'loss', 'losses')} across the {k} folds"
        if margin > se:
            reasons.append(
                f"It beat the runner-up, **{runner.name}** ({_fmt(runner.cv_mean[metric])}), by "
                f"**{_fmt(margin)}** — more than one standard error ({_fmt(se)}) — with {record} "
                "head-to-head (all models were scored on identical folds)."
            )
        else:
            tied = ", ".join(f"**{r.name}**" for r in contenders if r.name != top.name)
            reasons.append(
                f"It is statistically tied with {tied} (margin {_fmt(margin)}, below one standard error of "
                f"{_fmt(se)}; {record} against {runner.name}). It is kept because it has both the highest "
                "mean score and the simplest structure among the tied models."
            )

    # Which metrics does the chosen model lead on?
    leads = [
        METRIC_LABELS[m]
        for m in METRICS
        if chosen.cv_mean[m] >= max(r.cv_mean[m] for r in exp.models.values()) - 1e-12
    ]
    if leads:
        reasons.append(f"It has the best CV score on: {', '.join(leads)}.")

    holdout_best = max(exp.models.values(), key=lambda r: r.test_metrics[metric])
    if holdout_best.name == chosen.name:
        reasons.append(f"The hold-out set confirms the choice: it also has the best test {label}.")
    else:
        cautions.append(
            f"On the hold-out set **{holdout_best.name}** scored a higher {label} "
            f"({_fmt(holdout_best.test_metrics[metric])} vs {_fmt(chosen.test_metrics[metric])}). A single "
            "split is noisy, so the cross-validated ranking is trusted, but the gap is worth a second look."
        )

    reasons.append(
        f"Best hyperparameters ({chosen.n_candidates} candidates searched): {_describe_params(chosen.best_params)}."
    )
    reasons.append(MODEL_SPECS[chosen.name].description)

    gap = chosen.train_mean[metric] - chosen.cv_mean[metric]
    if gap > OVERFIT_GAP:
        cautions.append(
            f"Train-fold {label} ({_fmt(chosen.train_mean[metric])}) is {_fmt(gap)} above CV {label}: "
            "the model partly memorises training data. More data or stronger regularisation may help."
        )

    if chosen.cv_std[metric] > 0.05:
        cautions.append(
            f"Fold-to-fold spread is large (std {_fmt(chosen.cv_std[metric])}); results are sensitive to which "
            "rows land in training, which usually means the dataset is small."
        )

    if imbalance_ratio and imbalance_ratio >= 3:
        note = f"Classes are imbalanced ({imbalance_ratio:.1f}:1)."
        if metric == "accuracy":
            note += " Accuracy can look high just by predicting the majority class — F1 or ROC-AUC is a safer criterion."
        else:
            note += f" {label} is a sensible criterion here; check recall on the minority class in the confusion matrix."
        cautions.append(note)

    if not exp.is_binary:
        cautions.append(
            "This is a multiclass problem: precision, recall and F1 are macro-averaged (each class weighted "
            "equally) and ROC-AUC is one-vs-rest."
        )

    return Recommendation(
        model=chosen.name, top_scorer=top.name, used_one_se_rule=used_one_se, reasons=reasons, cautions=cautions
    )
