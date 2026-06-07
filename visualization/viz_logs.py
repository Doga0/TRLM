import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _normalize_key(key: str) -> str:
    """Normalize metric names like 'Lm_loss', 'Lmloss', 'Grad norm', 'GradNorm'."""
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _extract_metric_pairs(line: str) -> dict[str, float]:
    """Extract all `Metric: number` pairs from a log line."""
    metrics: dict[str, float] = {}
    for key, value in re.findall(rf"([A-Za-z_ ]+):\s*({FLOAT_RE})", line):
        metrics[_normalize_key(key)] = float(value)
    return metrics


def _extract_train_step(line: str) -> int | None:
    match = re.search(r"\bStep\s+(\d+)\b", line, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_eval_step(line: str) -> int | None:
    match = re.search(r"\bEval\s*Step\s+(\d+)\b", line, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def parse_log(file_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse TRDM training/evaluation logs.

    Supports both older labels:
      Lm_loss, Ppl, Grad norm, Eval Step, Masked PPL, Acc_Long

    and the labels in the provided logs:
      Lmloss, MaskedPerplexity, GradNorm, EvalStep, AccLong
    """
    train_data: list[dict[str, float | int]] = []
    eval_data: list[dict[str, float | int]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            metrics = _extract_metric_pairs(line)
            upper_line = line.upper()

            if "TRAIN" in upper_line:
                step = _extract_train_step(line)
                if step is None:
                    continue

                train_data.append({
                    "Step": step,
                    "Loss": metrics.get("loss"),
                    "Lm_loss": metrics.get("lmloss"),
                    "Ppl": metrics.get("ppl", metrics.get("maskedppl", metrics.get("maskedperplexity"))),
                    "Grad_norm": metrics.get("gradnorm"),
                    "Depth": metrics.get("depth"),
                    "Tokens": metrics.get("tokens"),
                })

            elif "EVAL" in upper_line:
                step = _extract_eval_step(line)
                if step is None:
                    continue

                eval_data.append({
                    "Step": step,
                    "Eval_Loss": metrics.get("loss"),
                    "Eval_PPL": metrics.get("ppl", metrics.get("maskedppl", metrics.get("maskedperplexity"))),
                    "Eval_Acc_Long": metrics.get("acclong"),
                    "Eval_Acc_Short": metrics.get("accshort"),
                    "Eval_Acc_Diff": metrics.get("accdiff"),
                    "Eval_Consistency": metrics.get("consistency"),
                    "Eval_Depth": metrics.get("depth"),
                })

    df_train = pd.DataFrame(train_data).dropna(how="all")
    df_eval = pd.DataFrame(eval_data).dropna(how="all")

    # Keep rows that have a step, but warn through stdout if some expected values were not parsed.
    if not df_train.empty:
        df_train = df_train.dropna(subset=["Step"]).sort_values("Step").reset_index(drop=True)
    if not df_eval.empty:
        df_eval = df_eval.dropna(subset=["Step"]).sort_values("Step").reset_index(drop=True)

    return df_train, df_eval


def _hide_unused_axes(axes, used_positions: set[tuple[int, int]]) -> None:
    for row_idx, row in enumerate(axes):
        for col_idx, axis in enumerate(row):
            if (row_idx, col_idx) not in used_positions:
                axis.set_visible(False)


def plot_metrics(df_train: pd.DataFrame, df_eval: pd.DataFrame, output_path: str | Path | None = None) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle("TRDM Pretraining Metrics", fontsize=20, fontweight="bold")
    used_positions: set[tuple[int, int]] = set()

    if not df_train.empty:
        axes[0, 0].plot(df_train["Step"], df_train["Loss"], label="Total Loss")
        axes[0, 0].plot(df_train["Step"], df_train["Lm_loss"], label="LM Loss", linestyle="--")
        axes[0, 0].set_title("Training Losses over Steps")
        axes[0, 0].set_xlabel("Steps")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        used_positions.add((0, 0))

        axes[0, 1].plot(df_train["Step"], df_train["Ppl"], label="Train Masked PPL")
        axes[0, 1].set_title("Training Masked Perplexity over Steps")
        axes[0, 1].set_xlabel("Steps")
        axes[0, 1].set_ylabel("Masked Perplexity")
        axes[0, 1].grid(True)
        used_positions.add((0, 1))

        axes[1, 0].plot(df_train["Step"], df_train["Tokens"], label="Tokens")
        axes[1, 0].set_title("Tokens Processed over Steps")
        axes[1, 0].set_xlabel("Steps")
        axes[1, 0].set_ylabel("Tokens")
        axes[1, 0].grid(True)
        used_positions.add((1, 0))

        axes[1, 1].plot(df_train["Step"], df_train["Grad_norm"], label="Grad Norm")
        axes[1, 1].set_title("Gradient Norm over Steps")
        axes[1, 1].set_xlabel("Steps")
        axes[1, 1].set_ylabel("Grad Norm")
        axes[1, 1].grid(True)
        used_positions.add((1, 1))

    if not df_eval.empty:
        axes[2, 0].plot(df_eval["Step"], df_eval["Eval_Loss"], label="Eval Loss", marker="o")
        axes[2, 0].set_title("Evaluation Loss over Steps")
        axes[2, 0].set_xlabel("Steps")
        axes[2, 0].set_ylabel("Eval Loss")
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        used_positions.add((2, 0))

        axes[2, 1].plot(df_eval["Step"], df_eval["Eval_PPL"], label="Masked PPL", marker="o")
        axes[2, 1].set_title("Evaluation Masked PPL over Steps")
        axes[2, 1].set_xlabel("Steps")
        axes[2, 1].set_ylabel("Eval Masked PPL")
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        used_positions.add((2, 1))

        axes[3, 0].plot(df_eval["Step"], df_eval["Eval_Acc_Long"], label="Acc Long", marker="o")
        axes[3, 0].plot(df_eval["Step"], df_eval["Eval_Acc_Short"], label="Acc Short", marker="x", linestyle="--")
        axes[3, 0].set_title("Evaluation Accuracy over Steps")
        axes[3, 0].set_xlabel("Steps")
        axes[3, 0].set_ylabel("Accuracy")
        axes[3, 0].legend()
        axes[3, 0].grid(True)
        used_positions.add((3, 0))

        axes[3, 1].plot(df_eval["Step"], df_eval["Eval_Consistency"], label="Eval Consistency", marker="o")
        axes[3, 1].set_title("Evaluation Consistency over Steps")
        axes[3, 1].set_xlabel("Steps")
        axes[3, 1].set_ylabel("Consistency")
        axes[3, 1].legend()
        axes[3, 1].grid(True)
        used_positions.add((3, 1))

    _hide_unused_axes(axes, used_positions)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and plot TRDM training/evaluation logs.")
    parser.add_argument(
        "log_file",
        nargs="?",
        default="training.log",
        help="Path to the training log file. Defaults to ./training.log",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output image path, e.g. metrics.png. If omitted, the plot window is shown.",
    )
    args = parser.parse_args()

    try:
        df_train, df_eval = parse_log(args.log_file)
    except FileNotFoundError:
        print(f"Error: log file not found: {args.log_file}")
        return

    print(f"Parsed {len(df_train)} training rows and {len(df_eval)} evaluation rows.")
    if df_train.empty and df_eval.empty:
        print("No matching log rows were found. Check whether the log format has changed again.")
        return

    plot_metrics(df_train, df_eval, output_path=args.save)


if __name__ == "__main__":
    main()
