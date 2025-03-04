import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Load data
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Load baseline model
model = joblib.load("models/baseline_model.pkl")

# LOO influence
np.random.seed(42)
selected_indices = np.random.choice(X_train.shape[0], 10, replace=False)
loo_influence = []
baseline_accuracy = accuracy_score(y_test, model.predict(X_test))

for idx in selected_indices:
    X_train_loo = X_train.drop(index=idx)
    y_train_loo = y_train.drop(index=idx)
    
    model_loo = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='liblinear')
    model_loo.fit(X_train_loo, y_train_loo)
    
    y_pred_loo = model_loo.predict(X_test)
    acc_loo = accuracy_score(y_test, y_pred_loo)
    loo_influence.append(acc_loo - baseline_accuracy)

for idx, influence in zip(selected_indices, loo_influence):
    print(f"Point {idx}: Influence = {influence:.4f}")

# Group-level influence
group_sizes = [int(X_train.shape[0] * p) for p in np.linspace(0.1, 0.5, 10)]
groups = [np.random.choice(X_train.shape[0], size, replace=False) for size in group_sizes]
group_influence = []

for group in groups:
    X_train_group = X_train.drop(index=group)
    y_train_group = y_train.drop(index=group)
    
    model_group = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='liblinear')
    model_group.fit(X_train_group, y_train_group)
    
    y_pred_group = model_group.predict(X_test)
    acc_group = accuracy_score(y_test, y_pred_group)
    group_influence.append(acc_group - baseline_accuracy)

for size, influence in zip(group_sizes, group_influence):
    print(f"Group size {size}: Influence = {influence:.4f}")

# Plot group size vs. influence
plt.plot(group_sizes, group_influence, marker='o')
plt.xlabel('Group Size')
plt.ylabel('Group Influence')
plt.title('Group Size vs. Influence')
plt.savefig("reports/groupsize_vs_influence.png")
plt.show()

# Shapley values
def compute_marginal_contribution(X_train, y_train, X_test, y_test, model, perm, i):
    subset = perm[:i+1]
    X_subset = X_train.iloc[subset]
    y_subset = y_train.iloc[subset]
    
    if len(np.unique(y_subset)) < 2:
        return 0
    
    model.fit(X_subset, y_subset)
    acc_with = accuracy_score(y_test, model.predict(X_test))
    
    subset_without = perm[:i]
    X_subset_without = X_train.iloc[subset_without]
    y_subset_without = y_train.iloc[subset_without]
    
    if len(np.unique(y_subset_without)) < 2:
        return 0
    
    model.fit(X_subset_without, y_subset_without)
    acc_without = accuracy_score(y_test, model.predict(X_test))
    
    return acc_with - acc_without

def truncated_mc_shapley(X_train, y_train, X_test, y_test, model, n_permutations=10):
    n = X_train.shape[0]
    shapley_values = np.zeros(n)
    
    results = Parallel(n_jobs=-1)(
        delayed(compute_marginal_contribution)(X_train, y_train, X_test, y_test, model, np.random.permutation(n), i)
        for _ in range(n_permutations)
        for i in range(n)
    )
    
    results = np.array(results).reshape(n_permutations, n)
    shapley_values = results.mean(axis=0)
    
    return shapley_values

# Subsample for Shapley values
np.random.seed(42)
sample_indices = np.random.choice(X_train.shape[0], 1000, replace=False)
X_train_sampled = X_train.iloc[sample_indices]
y_train_sampled = y_train.iloc[sample_indices]

# Compute Shapley values
shapley_values = truncated_mc_shapley(
    X_train_sampled, y_train_sampled, X_test, y_test, 
    LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='liblinear')
)

# Plot Shapley values
plt.hist(shapley_values, bins=20)
plt.xlabel('Shapley Value')
plt.ylabel('Frequency')
plt.title('Distribution of Shapley Values')
plt.savefig("reports/shapley_distribution.png")
plt.show()