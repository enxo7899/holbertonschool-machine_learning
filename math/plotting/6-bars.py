#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fig, ax = plt.subplots()
for i in range(len(fruit)):
    ax.bar(np.arange(len(fruit[i])) + 0.25, fruit[i], width=0.5, bottom=np.sum(fruit[:i], axis=0), color=colors[i])
ax.legend(['apples', 'bananas', 'oranges', 'peaches'])
ax.set_ylabel('Quantity of Fruit', fontsize='x-small')
ax.set_xlabel('Person', fontsize='x-small')
ax.set_title('Number of Fruit per Person', fontsize='x-small')
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))
ax.set_xticks(np.arange(len(fruit[0])) + 0.25)
ax.set_xticklabels(['Farrah', 'Fred', 'Felicia'], fontsize='x-small')
plt.show()
        
