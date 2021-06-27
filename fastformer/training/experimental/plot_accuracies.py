import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
import numpy as np
import random
sns.set()

accuracies = pd.concat([pd.DataFrame({"layer": list(range(1, 13))*2, "accuracy": (torch.sigmoid(torch.rand(12)) * 100).tolist() + (torch.sigmoid(torch.rand(12)) * 50).tolist(), "center": (["with center"] * 12) + (["after center removal"] * 12)}) for _ in range(5)])
fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
sns.lineplot(data=accuracies, x="layer", y="accuracy", hue="center", ax=axes, markers=True, err_style="band")
# axes.set_title("Layer Wise accuracy with and without center removal")
fig.suptitle("Layer Wise accuracy of a linear classifier with and without center removal")
plt.subplots_adjust(top=0.9, bottom=0.09,)
plt.ylim(0, 100)
plt.show()

