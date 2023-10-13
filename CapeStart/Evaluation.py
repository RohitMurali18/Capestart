from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from Tuning import best_params
from Training import X_train, y_train


clf = LogisticRegression(C=best_params['C'], max_iter=1000, random_state=42)


scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')  # You can adjust the number of folds (cv)

for i, score in enumerate(scores):
    print(f"Fold {i+1} Accuracy: {score:.2f}")


mean_accuracy = scores.mean()
std_accuracy = scores.std()
print(f"Mean Accuracy: {mean_accuracy:.2f}")
print(f"Standard Deviation: {std_accuracy:.2f}")