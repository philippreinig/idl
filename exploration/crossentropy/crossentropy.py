import numpy as np
import matplotlib.pyplot as plt


def cross_entropy(p: list[float], q: list[float]):
    if len(p) != len(q):
        raise ValueError("length of p and q have to be equal!")

    n = len(p)
    sum = 0
    for i in range(n):
        sum += - p[i] * np.log(q[i])

    return sum

def binary_cross_entropy(p, q):
    if q == 1:
        return float('NaN')
    return -(p * np.log(q) + (1 - p) * np.log(1 - q))


p = [round(x, 1) for x in np.linspace(0.1, 1, 10)]
q = [round(x, 2) for x in np.linspace(0.1, 1, 10)]

table = np.zeros((len(p), len(q)))

for i in range(len(q)):
    for j in range(len(p)):
        bce = round(binary_cross_entropy(p[i], q[j]), 3)
        table[j][i] = bce
        print(f"bce(q={q[i]}, p={p[j]})={bce}")

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(table, cmap="Blues")

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(p)), labels=p, fontsize=15)
ax.set_yticks(np.arange(len(q)), labels=q, fontsize=15)

ax.xaxis.tick_top()


ax.set_xlabel("Actual probability p", fontsize=15, labelpad=20)
ax.set_ylabel("Predicted probability q", fontsize=15, labelpad=20)

ax.xaxis.set_label_position('top')

print(table)

# Loop over data dimensions and create text annotations.
for i in range(len(q)):
    for j in range(len(p)):
        color_val = "white" if table[i, j] >= 1.5 else "black"
        cell_val = table[i, j]

        if i == len(q) - 1:
            cell_val = "Undefined!"[j]

        print(f"{table[i,j]}, {cell_val}, {color_val}")

        text = ax.text(j, i, cell_val,
                       ha="center", va="center", color=color_val, fontsize=14)

#ax.set_title("Binary Cross-Entropy", fontsize=30, pad=20)
fig.tight_layout()

plt.savefig("bce.png")

plt.show()

print()
