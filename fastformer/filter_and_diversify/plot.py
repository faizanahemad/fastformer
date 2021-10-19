import pandas as pd
import numpy as np
import sys

argv = list(sys.argv)

from IPython.terminal.embed import InteractiveShellEmbed
my_shell = InteractiveShellEmbed()
result = my_shell.getoutput("pip install uniplot")
print(result)
result = my_shell.getoutput("cat output.log | grep %s | awk '{print $NF}'" % (argv[1]))

from uniplot import plot
result = list(map(float, result))
result = pd.DataFrame({0:result}).ewm(alpha=float(argv[2])).mean().fillna(0.0)[0].clip(0.0, 1e9)
plot(result)
