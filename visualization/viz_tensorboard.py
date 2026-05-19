import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_events(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    return event_acc


def get_scalar(event_acc, tag):
    if tag not in event_acc.Tags()["scalars"]:
        print(f"[WARN] Tag not found: {tag}")
        return [], []

    events = event_acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def plot_metric(event_acc, tag):
    steps, values = get_scalar(event_acc, tag)
    if not steps:
        return

    plt.figure()
    plt.plot(steps, values)
    plt.title(tag)
    plt.xlabel("Step")
    plt.ylabel(tag)
    plt.grid()
    plt.show()


def plot_group(event_acc, tags, title):
    plt.figure()
    for tag in tags:
        steps, values = get_scalar(event_acc, tag)
        if steps:
            plt.plot(steps, values, label=tag)

    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    log_dir = "C:/Users/USER/Desktop/DOA/projects/bitirme/TRDLM/logs/2/events.out.tfevents.1778687791.aed2ebfce299.693.0" 

    event_acc = load_events(log_dir)

    # Train metrics
    train_tags = [
        "Train/Loss",
        "Train/LmLoss",
        "Train/ShortLmLoss",
        "Train/ConsistencyLoss"
        "Train/MaskedPerplexity",
        "Train/LearningRate"
    ]

    plot_group(event_acc, train_tags, "Train Metrics")

    # Eval metrics
    eval_tags = [
        "Eval/Loss",
        "Eval/Masked_Perplexity",
        "Eval/Acc_Long",
        "Eval/Acc_Short",
        "Eval/Acc_Diff",
        "Eval/Consistency"
        "Eval/Depth",
    ]

    plot_group(event_acc, eval_tags, "Eval Metrics")

    # Other metrics
    extra_tags = [
        "Metrics/RecursionDepth",
        "Metrics/TokensSeen",
        "Metrics/GradNorm"
    ]

    plot_group(event_acc, extra_tags, "Other Metrics")