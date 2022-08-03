def evaluate_model(model, loss_func, X_test, y_test):
	score = 0
	for X, y in zip(X_test, y_test):
		out = model.forward(X)
		loss = loss_func(y_test, out)
		score += loss
	score /= len(X_test)
	return score