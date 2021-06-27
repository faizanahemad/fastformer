import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
import numpy as np
import random
sns.set()

accuracies = pd.concat([pd.DataFrame({"layer": list(range(1, 13))*2, "L2-Norm": (torch.arange(12) * 0.5 + torch.sigmoid(torch.rand(12)) * 5).tolist() + (torch.sigmoid(torch.rand(12)) * 5).tolist(), "Output location": (["X_out"] * 12) + (["TRANSFORM(X)"] * 12)}) for _ in range(5)])
fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
sns.lineplot(data=accuracies, x="layer", y="L2-Norm", hue="Output location", ax=axes, markers=True, err_style="band")
# axes.set_title("Layer Wise accuracy with and without center removal")
fig.suptitle("Layer Wise average token-wise L2-Norm")
plt.subplots_adjust(top=0.9, bottom=0.09,)
plt.ylim(0, 10)
plt.show()

