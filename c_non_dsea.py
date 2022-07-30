classifier.fit(X_train, y_train)
y_test_pred = classifier.predict_proba(X_test)
f_test_pred = np.bincount(y_test) / wandb.config.num_bins

probas = y_test_pred


dist = util.chi2s(f_test_true, f_test_pred)
print(f"Chi2 distance: {dist:.10f}")
