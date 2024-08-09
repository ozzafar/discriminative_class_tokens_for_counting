import matplotlib.pyplot as plt
import numpy as np

# Example data
Q1 = np.array([1, 1, 0.692307692, 1, 0.846153846, 0.769230769, 0.846153846, 1, 0.923076923, 0.769230769, 0.923076923, 1, 0.923076923, 0.615384615, 1, 1, 0.923076923, 0.692307692, 1, 0.846153846, 0.692307692, 0.923076923])
Q2 = np.array([1, 0.692307692, 0.615384615, 1, 0.846153846, 0.846153846, 0.923076923, 0.923076923, 0.846153846, 0.923076923, 0.769230769, 0.923076923, 0.615384615, 0.461538462, 1, 0.769230769, 0.846153846, 0.923076923, 0.615384615, 0.615384615, 0.461538462, 0.923076923])

# Calculate mean and std
mean1, std1 = np.mean(Q1), np.std(Q1)
mean2, std2 = np.mean(Q2), np.std(Q2)

print(f"Mean: Q1 - {mean1}, Q1 - {mean2}")

# Create a barplot with means and error bars for std
plt.figure(figsize=(10, 6))
plt.bar(["Which image best matches the number\nof objects of the prompt?", "Which image is more natural?"], [mean1, mean2], yerr=[std1, std2], color=['blue', 'orange'], capsize=5, edgecolor='black')
plt.title('Human Evaluation', fontsize=12)
plt.ylabel("Mean score")
plt.xticks(fontsize=12)

# Show the plot
plt.savefig('human_study.png', dpi=300, bbox_inches='tight')
plt.show()