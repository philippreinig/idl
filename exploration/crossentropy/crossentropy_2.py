import numpy as np
import matplotlib.pyplot as plt

def binary_cross_entropy(o, y):
    return -y * np.log(o) - (1 - y) * np.log(1 - o)

o = np.linspace(0.01, 0.99, 100)
y = [1.0]

plt.rcParams.update({'font.size': 15})

fig, ax = plt.subplots(figsize=(13, 10))
for x in y:
    line_color = "blue"
    # if x == 1:
    #     plot_color = "red"
    ax.plot(o, [binary_cross_entropy(i, x) for i in o], label=f"p = {x:.2f}", color=line_color)

ax.set_xlabel("Predicted Probability of the Model", labelpad=10)
ax.set_ylabel("Binary Cross Entropy Loss", labelpad=10)
# ax.set_title("Binary Cross Entropy Loss for different combinations of ground truth label p and predicted probability of the model")

ax.legend()

fig.tight_layout()

plt.savefig("bce3.png")

plt.show()