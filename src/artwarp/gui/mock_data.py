"""Deterministic fake metrics for a simulated training run.

Everything comes from the run's seed. The finished chart specs are also
stored in run.json, so old runs don't depend on this module staying the same.

Chart specs are plain dicts (see widgets/chart.py for how each kind is drawn).
Series "slot" picks the colour: 0 = train, 1 = validation, 2 = other.
"""
import math
import random

CLASSES = ["cat", "dog", "car", "ship", "bird", "frog"]
TRAIN, VAL, OTHER = 0, 1, 2

CURVE_KEYS = (
    "train_loss", "val_loss", "train_acc", "val_acc", "top5", "precision", "recall",
    "f1", "auc", "lr", "grad_norm", "epoch_time", "gpu_mem", "throughput",
)


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def make_curves(seed, epochs):
    """Per-epoch metric curves. Live training reads these one epoch at a time."""
    rng = random.Random(seed)
    start, floor = rng.uniform(2.0, 2.4), rng.uniform(0.06, 0.14)
    rate, gap, overfit = rng.uniform(3.5, 5.0), rng.uniform(0.04, 0.1), rng.uniform(0.6, 1.4)
    cap = rng.uniform(0.88, 0.95)
    epoch_time, warm = rng.uniform(18, 26), max(1, round(epochs * 0.1))
    c = {k: [] for k in CURVE_KEYS}

    for i in range(epochs):
        t = i / max(epochs - 1, 1)
        base = floor + (start - floor) * math.exp(-rate * t)
        prog = 1 - math.exp(-rate * 1.05 * t)

        c["train_loss"].append(base * (1 + rng.gauss(0, 0.025)))
        c["val_loss"].append((base + gap + overfit * max(0.0, t - 0.6) ** 2) * (1 + rng.gauss(0, 0.035)))
        c["train_acc"].append(_clamp(0.1 + (cap + 0.04 - 0.1) * prog + rng.gauss(0, 0.006)))
        va = _clamp(0.1 + (cap - 0.1) * prog - 0.05 * max(0.0, t - 0.75) + rng.gauss(0, 0.009))
        c["val_acc"].append(va)
        c["top5"].append(_clamp(0.5 + 0.49 * (1 - math.exp(-rate * 1.6 * t)) + rng.gauss(0, 0.004), hi=0.998))

        p = _clamp(va + 0.012 + rng.gauss(0, 0.008))
        r = _clamp(va - 0.01 + rng.gauss(0, 0.008))
        c["precision"].append(p)
        c["recall"].append(r)
        c["f1"].append(2 * p * r / (p + r) if p + r else 0.0)
        c["auc"].append(_clamp(0.5 + 0.49 * (1 - math.exp(-rate * 1.4 * t)) + rng.gauss(0, 0.003)))

        # linear warm-up, then cosine decay
        if i < warm:
            lr = 1e-3 * (i + 1) / warm
        else:
            lr = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * (i - warm) / max(epochs - warm - 1, 1)))
        c["lr"].append(lr)

        spike = rng.uniform(0.8, 1.6) if rng.random() < 0.08 else 0.0
        c["grad_norm"].append(0.6 + 3.2 * math.exp(-3.5 * t) + abs(rng.gauss(0, 0.15)) + spike)
        c["epoch_time"].append(epoch_time + (9 if i == 0 else 0) + rng.gauss(0, 0.7))
        c["gpu_mem"].append(6.8 + 0.6 * (1 - math.exp(-8 * t)) + rng.gauss(0, 0.03))
        c["throughput"].append(2400 - (550 if i == 0 else 0) + rng.gauss(0, 35))
    return c


def summarise(c):
    return {
        "final_train_loss": c["train_loss"][-1],
        "final_val_loss": c["val_loss"][-1],
        "best_val_acc": max(c["val_acc"]),
    }


def build_charts(seed, epochs, c):
    """The 20 result charts, in grid order (4 rows x 5 columns)."""
    rng = random.Random(seed + 1)
    x = list(range(1, epochs + 1))
    final_acc = c["val_acc"][-1]

    def series(name, values, slot):
        return {"name": name, "slot": slot, "values": [round(v, 6) for v in values]}

    def line(title, desc, lines, y_label, fmt="float", **extra):
        return {"kind": "line", "title": title, "desc": desc, "x": x, "x_label": "Epoch",
                "y_label": y_label, "y_format": fmt, "series": lines, **extra}

    def bar(title, desc, cats, values, slot, x_label, y_label, fmt):
        return {"kind": "bar", "title": title, "desc": desc, "categories": cats,
                "x_label": x_label, "y_label": y_label, "y_format": fmt,
                "series": [series(y_label, values, slot)]}

    def hist(title, desc, samples, lo, hi, bins=24):
        width = (hi - lo) / bins
        counts = [0] * bins
        for v in samples:
            k = int((v - lo) / width)
            if 0 <= k < bins:
                counts[k] += 1
        return {"kind": "hist", "title": title, "desc": desc, "slot": TRAIN,
                "edges": [round(lo + k * width, 6) for k in range(bins + 1)], "values": counts,
                "x_label": "Weight value", "y_label": "Count", "y_format": "int", "x_format": "float"}

    offsets = {"cat": -0.06, "dog": -0.07, "car": 0.04, "ship": 0.03, "bird": -0.02, "frog": 0.02}
    per_class = [_clamp(final_acc + offsets[k] + rng.gauss(0, 0.01)) for k in CLASSES]

    confusion = []
    for i, name in enumerate(CLASSES):
        correct = round(500 * per_class[i])
        weights = [0.0 if j == i else rng.uniform(0.3, 1.0) * (5 if {name, other} == {"cat", "dog"} else 1)
                   for j, other in enumerate(CLASSES)]
        row = [round((500 - correct) * w / sum(weights)) for w in weights]
        row[i] = correct
        confusion.append(row)

    conf = [0.05 + 0.1 * k for k in range(10)]
    calibration = [[round(cf, 3), round(_clamp(cf - 0.08 * math.sin(math.pi * cf) + rng.gauss(0, 0.015)), 4)]
                   for cf in conf]

    centres = {"cat": (-2.0, 1.0), "dog": (-1.1, 1.9), "car": (2.2, -1.2)}
    embedding = [
        {"name": name, "slot": slot,
         "points": [[round(cx + rng.gauss(0, 0.6), 3), round(cy + rng.gauss(0, 0.6), 3)] for _ in range(60)]}
        for slot, (name, (cx, cy)) in enumerate(centres.items())
    ]

    w_first = [rng.gauss(0, 0.045) for _ in range(4000)]
    w_out = [rng.gauss(0.01, 0.09) + (rng.gauss(0.15, 0.05) if rng.random() < 0.15 else 0) for _ in range(4000)]

    return [
        line("Loss", "Cross-entropy on the train and validation sets.",
             [series("Train", c["train_loss"], TRAIN), series("Validation", c["val_loss"], VAL)], "Cross-entropy"),
        line("Accuracy", "Top-1 accuracy per epoch.",
             [series("Train", c["train_acc"], TRAIN), series("Validation", c["val_acc"], VAL)], "Accuracy", "pct"),
        line("Learning rate", "Linear warm-up followed by cosine decay.",
             [series("Learning rate", c["lr"], OTHER)], "Learning rate", "sci", y_zero=True),
        line("Gradient norm", "Global L2 norm of the gradients, averaged per epoch.",
             [series("Grad norm", c["grad_norm"], TRAIN)], "L2 norm", y_zero=True),
        line("Top-5 accuracy", "Validation set.", [series("Validation", c["top5"], VAL)], "Top-5 accuracy", "pct"),

        line("Precision", "Macro-averaged, validation set.", [series("Validation", c["precision"], VAL)], "Precision", "pct"),
        line("Recall", "Macro-averaged, validation set.", [series("Validation", c["recall"], VAL)], "Recall", "pct"),
        line("F1 score", "Macro-averaged, validation set.", [series("Validation", c["f1"], VAL)], "F1", "pct"),
        line("ROC AUC", "One-vs-rest, macro-averaged, validation set.", [series("Validation", c["auc"], VAL)], "AUC"),
        bar("Epoch time", "Wall-clock seconds per epoch. Epoch 1 includes warm-up.",
            [str(e) for e in x], c["epoch_time"], OTHER, "Epoch", "Seconds", "float"),

        bar("Per-class accuracy", "Validation accuracy for each class at the final epoch.",
            CLASSES, per_class, VAL, "Class", "Accuracy", "pct"),
        bar("Class distribution", "Training samples per class.",
            CLASSES, [rng.randint(4200, 5800) for _ in CLASSES], TRAIN, "Class", "Samples", "int"),
        {"kind": "heatmap", "title": "Confusion matrix", "desc": "Validation set, final epoch. Rows are true labels.",
         "labels": CLASSES, "matrix": confusion, "x_label": "Predicted", "y_label": "True"},
        {"kind": "scatter", "title": "Calibration", "desc": "Mean confidence vs observed accuracy, 10 bins. Diagonal = perfect.",
         "series": [{"name": "Validation", "slot": VAL, "points": calibration}], "diagonal": True,
         "x_label": "Predicted confidence", "y_label": "Observed accuracy", "x_format": "pct", "y_format": "pct"},
        {"kind": "scatter", "title": "Embedding (2D)", "desc": "Penultimate-layer features projected to 2D, three classes.",
         "series": embedding, "x_label": "Component 1", "y_label": "Component 2", "x_format": "float", "y_format": "float"},

        hist("Weights: first layer", "Distribution of first-layer weights at the end of training.", w_first, -0.2, 0.2),
        hist("Weights: output layer", "Distribution of output-layer weights at the end of training.", w_out, -0.35, 0.45),
        bar("Activation sparsity", "Share of zero activations per layer (ReLU).",
            [f"L{i}" for i in range(1, 9)], [_clamp(0.12 + 0.065 * i + rng.gauss(0, 0.02)) for i in range(8)],
            OTHER, "Layer", "Zero activations", "pct"),
        line("GPU memory", "Peak allocated memory per epoch.", [series("GPU memory", c["gpu_mem"], OTHER)], "GB"),
        line("Throughput", "Training samples processed per second.",
             [series("Throughput", c["throughput"], OTHER)], "Samples / s"),
    ]
