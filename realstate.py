# i use a machine learning model (Random forest) for this dataset


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n🔁 K-Fold Cross-Validation:")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model_kf = RandomForestClassifier()
kf_scores = cross_val_score(model_kf, X, y, cv=kf)
print("Scores:", kf_scores)
print("Average Accuracy (KFold): {:.2f}%".format(kf_scores.mean() * 100))

print("\n🔁 StratifiedKFold Cross-Validation:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_skf = RandomForestClassifier()
skf_scores = cross_val_score(model_skf, X, y, cv=skf)
print("Scores:", skf_scores)
print("Average Accuracy (StratifiedKFold): {:.2f}%".format(skf_scores.mean() * 100))

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

