import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import random
sns.set()

data = np.random.randn(6, 32)
fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey=True)
axes = [axb for ax in axes for axb in ax]
freq = 8
for i in range(len(data)):
    data[i] = data[i] + random.randint(-1, 1) * random.random()
    data[i] = data[i] * (((i + 1)*2) ** 0.5)
    x = list(range(len(data[i])))
    data[i][random.randint(0, len(x) - 1)] += (np.sign(data[i][random.randint(0, len(x) - 1)]) * random.random() * (((i + 1)*2) ** 0.5))
    sns.barplot(ax=axes[i], x=x, y=list(data[i]))
    axes[i].set_title("Layer %s" % ((i + 1) * 2))
    xtix = axes[i].get_xticks()
    axes[i].set_xticks(xtix[::freq])
    axes[i].set_xticklabels(x[::freq])

fig.suptitle("Layer-wise feature centers for 1st 32 dimensions of d")
plt.subplots_adjust(top=0.9, bottom=0.09,)
plt.show()
