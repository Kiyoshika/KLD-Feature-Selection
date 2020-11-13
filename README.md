# KLD-Feature-Selection
A feature selection algorithm I wrote based on KL Divergence. Currently only supports binary classification problems but will expand in the future.

# How To Use
* Let's make a fake data set using sklearn's make_classification
```python
x_full, y_full = make_classification(n_samples = 10000, n_features = 200, n_informative = 16, n_redundant = 8)
x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size = 0.3)
```
* We can now use the kld_feature_selection() function to return the indices that are separable between classes
```python
relevant_features = kld_feature_selection(x_train, y_train) 
relevant_features # array([ 20,  24,  29,  47,  64,  79,  83, 121, 123, 143, 188, 190])
```
* The parameters of this function are as follows:
  * feature_set: An np.array containing your independent variables
  * target_set: An np.array containing your output variable (currently only supports 0 and 1 for binary problems)
  * threshold: An optional parameter indicating the number of standard deviations the KL divergence of a feature must pass in order to keep; default value is 1.
    * KL divergence scores are standard-scaled in order to compare against each other dynamically instead of using static values.
* Building two logistic regression models with the same seed to compare 5-fold CV score and test f1 score:
```python
lr_full = LogisticRegression(max_iter=2500, random_state = 211)
lr_full_cv = cross_val_score(lr_full, x_train, y_train, cv=5, scoring='f1')

lr_full.fit(x_train, y_train)
lr_full_pred = lr_full.predict(x_test)
print("5-CV Score: ", lr_full_cv.mean()) # 0.8287365414774207
print("Test F1 Score: ", f1_score(y_test, lr_full_pred)) # 0.8209606986899562
```

```python
lr_reduced = LogisticRegression(max_iter=2500, random_state = 211)
lr_reduced_cv = cross_val_score(lr_reduced, x_train[:,relevant_features], y_train, cv=5, scoring='f1')

lr_reduced.fit(x_train[:,relevant_features], y_train)
lr_reduced_pred = lr_reduced.predict(x_test[:,relevant_features])
print("5-CV Score: ", lr_reduced_cv.mean()) # 0.8172785607745185
print("Test F1 Score: ", f1_score(y_test, lr_reduced_pred)) # 0.8103620059780804
```
* Our reduced model isn't **quite** as good, however we achieved very similar performance while dropping 188 features! Significantly less complex and easier to interpret.
