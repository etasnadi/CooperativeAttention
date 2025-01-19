import matplotlib.pyplot as plt
import math
from collections import defaultdict

magic = '#EXPERIMENT#'

groupby = lambda g: (g[0] + '-' + g[1] + '-' + g[3], int(g[2][1:]), g[4])

dataset = defaultdict(list)
with open('perf_data.log', 'r') as exp_file:
    for line_raw in exp_file:
        line = line_raw.strip()
        if magic in line:
            data = line[len(magic):].split(';')
            group, label, value = groupby(data)
            dataset[group].append((label, value))

colors = ['blue', 'red', 'green', 'yellow']

maximum = -1
for idx, (diag, contents) in enumerate(dataset.items()):
    col = colors[idx % len(colors)]
    
    a_values = list(range(len(contents)))
    a_labels = list(c[0] for c in contents)
    b = list(float(c[1]) for c in contents)
    maximum = max(max(b), maximum)
    plt.plot(a_values, b, color=col, label=diag, marker='o', markersize=3)

num_of_ticks = 5
plt.xlabel('sequence length')
plt.ylabel('TFLOPS')
plt.xticks(a_values, labels=a_labels)
plt.yticks(range(0, math.ceil(maximum), round(maximum/(num_of_ticks+1))))
plt.legend()
plt.show();
