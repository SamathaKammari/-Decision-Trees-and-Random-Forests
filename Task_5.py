#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt


# In[2]:


pip install graphviz


# In[8]:


# Load the data
data = pd.read_csv('breast-cancer-wisconsin.data', header=None,
                   names=['ID', 'Clump_Thickness', 'Cell_Size_Uniformity', 'Cell_Shape_Uniformity',
                          'Marginal_Adhesion', 'Epithelial_Cell_Size', 'Bare_Nuclei',
                          'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class'])


# In[10]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


# Drop ID column and prepare features and target
X = data.drop(['ID', 'Class'], axis=1)
y = data['Class'].map({2: 0, 4: 1})  # Convert 2->0 (benign), 4->1 (malignant)


# In[31]:


from sklearn.tree import export_text

# Print text-based tree
tree_text = export_text(dt, feature_names=list(feature_names))
print("Decision Tree Structure:\n")
print(tree_text)


# In[13]:


# Handle missing values ('?' in Bare_Nuclei)
data['Bare_Nuclei'] = data['Bare_Nuclei'].replace('?', np.nan)
# Replace missing values with the mode of the column
data['Bare_Nuclei'] = data['Bare_Nuclei'].fillna(data['Bare_Nuclei'].mode()[0]).astype(int)


# In[32]:


# Drop ID column and prepare features and target
X = data.drop(['ID', 'Class'], axis=1)
y = data['Class'].map({2: 0, 4: 1})  # Convert 2->0 (benign), 4->1 (malignant)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[33]:


# Step 1: Train and Visualize Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)


# In[34]:


# Predict and evaluate
y_pred_dt = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")


# In[35]:


# Step 2: Analyze Overfitting and Control Tree Depth
dt_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_pruned.fit(X_train, y_train)


# In[36]:


# Evaluate pruned tree
y_pred_pruned = dt_pruned.predict(X_test)
pruned_accuracy = accuracy_score(y_test, y_pred_pruned)
print(f"Pruned Decision Tree Accuracy (max_depth=3): {pruned_accuracy:.4f}")


# In[37]:


# Compare train vs. test accuracy
train_acc = accuracy_score(y_train, dt.predict(X_train))
test_acc = accuracy_score(y_test, dt.predict(X_test))
print(f"Default Tree - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
train_acc_pruned = accuracy_score(y_train, dt_pruned.predict(X_train))
test_acc_pruned = accuracy_score(y_test, dt_pruned.predict(X_test))
print(f"Pruned Tree - Train Accuracy: {train_acc_pruned:.4f}, Test Accuracy: {test_acc_pruned:.4f}")


# In[38]:


# Step 3: Train Random Forest and Compare Accuracy
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


# In[39]:


# Predict and evaluate
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")


# In[40]:


# Step 4: Interpret Feature Importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]


# In[41]:


# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()


# In[42]:


# Step 5: Evaluate Using Cross-Validation
dt_cv_scores = cross_val_score(dt, X, y, cv=5)
rf_cv_scores = cross_val_score(rf, X, y, cv=5)

print(f"Decision Tree CV Accuracy: {np.mean(dt_cv_scores):.4f} ± {np.std(dt_cv_scores):.4f}")
print(f"Random Forest CV Accuracy: {np.mean(rf_cv_scores):.4f} ± {np.std(rf_cv_scores):.4f}")


# In[29]:


# Save DOT file
with open("decision_tree_breast_cancer.dot", "w") as f:
    f.write(dot_data)
print("DOT file saved as 'decision_tree_breast_cancer.dot'. Use an online viewer or install Graphviz to render.")


# In[ ]:




