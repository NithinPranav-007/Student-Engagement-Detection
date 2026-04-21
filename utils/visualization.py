from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
