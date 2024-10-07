import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = {
    "8b": {
        "inv_sub": {"Correct (w Neg sign)": 240, "Correct (Neg sign Ignored)": 435, "Incorrect": 309},
        "inv_add": {"Correct (w Neg sign)": 355, "Correct (Neg sign Ignored)": 219, "Incorrect": 410},
        "oov_sub": {"Correct (w Neg sign)": 134, "Correct (Neg sign Ignored)": 151, "Incorrect": 751},
        "oov_add": {"Correct (w Neg sign)": 342, "Correct (Neg sign Ignored)": 9, "Incorrect": 685}
    },
    "70b": {
        "inv_sub": {"Correct (w Neg sign)": 25, "Correct (Neg sign Ignored)": 927, "Incorrect": 32},
        "inv_add": {"Correct (w Neg sign)": 33, "Correct (Neg sign Ignored)": 844, "Incorrect": 107},
        "oov_sub": {"Correct (w Neg sign)": 52, "Correct (Neg sign Ignored)": 589, "Incorrect": 395},
        "oov_add": {"Correct (w Neg sign)": 114, "Correct (Neg sign Ignored)": 746, "Incorrect": 176}
    }
}
total_inv = sum(data['8b']['inv_add'].values())
total_oov = sum(data['8b']['oov_add'].values())

categories = ["Correct_with_neg_sign", "Correct_w/o_neg_sign", "Incorrect"]
x = np.arange(len(categories))
width = 0.3

def plot_bars(ax, data1, data2, label1, label2, title):
    rects1 = ax.bar(x - width/2, data1, width, label=label1, color='skyblue')
    rects2 = ax.bar(x + width/2, data2, width, label=label2, color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1000)
    ax.legend()
    ax.set_title(title)
    for rect in rects1 + rects2:
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 10, int(rect.get_height()), ha='center')

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plotting 8b vs 70b comparisons 
plot_bars(axs[0, 0], list(data["8b"]["inv_add"].values()), list(data["70b"]["inv_add"].values()), 
          '8b Inv', '70b Inv', 'In-vocab (Query:Sub, Context: Add): 8b vs 70b')
plot_bars(axs[0, 1], list(data["8b"]["oov_add"].values()), list(data["70b"]["oov_add"].values()), 
          '8b OOV', '70b OOV', 'OOV  (Query:Sub, Context: Add): 8b vs 70b')

plot_bars(axs[1, 0], list(data["8b"]["inv_sub"].values()), list(data["70b"]["inv_sub"].values()), 
          '8b Inv', '70b Inv', 'In-vocab (Query:Sub, Context: Sub): 8b vs 70b')
plot_bars(axs[1, 1], list(data["8b"]["oov_sub"].values()), list(data["70b"]["oov_sub"].values()), 
          '8b OOV', '70b OOV', 'OOV (Query:Sub, Context: Sub): 8b vs 70b')

plt.suptitle(f"llama-3.1-8b and llama-3.1-70b for Subtraction (INV = {total_inv}, OOV = {total_oov})", fontsize=16)
plt.tight_layout()
#plt.show()
fig.savefig("understand_llm_math/exploration/std_op/combined_figures/8b_vs_70b_comparison.pdf")

# Plotting 8b comparisons for inv vs oov
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plot_bars(axs[0, 0], list(data["8b"]["inv_add"].values()), list(data["8b"]["oov_add"].values()), 
          '8b Inv', '8b OOV', '8b Inv vs OOV: (Query:Sub, Context: Add)')
plot_bars(axs[1, 0], list(data["8b"]["inv_sub"].values()), list(data["8b"]["oov_sub"].values()), 
          '8b Inv', '8b OOV', '8b Inv vs OOV: (Query:Sub, Context: Sub)')

# Plotting 70b comparisons for inv vs oov
plot_bars(axs[0, 1], list(data["70b"]["inv_add"].values()), list(data["70b"]["oov_add"].values()), 
          '70b Inv', '70b OOV', '70b Inv vs OOV: (Query:Sub, Context: Add)')
plot_bars(axs[1, 1], list(data["70b"]["inv_sub"].values()), list(data["70b"]["oov_sub"].values()), 
          '70b Inv', '70b OOV', '70b Inv vs OOV: (Query:Sub, Context: Sub)')

plt.suptitle(f"llama-3.1-8b and llama-3.1-70b for Subtraction (INV = {total_inv}, OOV = {total_oov})", fontsize=16)
plt.tight_layout()
fig.savefig("understand_llm_math/exploration/std_op/combined_figures/inv_vs_oov_comparison.pdf")
#plt.show()
