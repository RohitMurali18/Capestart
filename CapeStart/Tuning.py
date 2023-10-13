from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from Training import X_train, y_train,accuracy_score,classification_report,X_test,y_test


param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
}


clf = LogisticRegression(max_iter=1000, random_state=42)


grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)


grid_search.fit(X_train, y_train)


best_params = grid_search.best_params_


best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)