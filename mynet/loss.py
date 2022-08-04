def sse_loss(targets, predictions):
    sum_sq_err = 0
    for target, pred in zip(targets, predictions):
        for t, p in zip(target, pred):
            sum_sq_err += (t - p) ** 2
    return sum_sq_err


def mse_loss(targets, predictions):
    return sse_loss(targets, predictions) / len(targets)


def cross_entropy_loss(targets, predictions, epsilon=1e-10):
    log_sum = 0
    for target, pred in zip(targets, predictions):
        target_idx = target.index(max(target))
        log_sum += (pred[target_idx] + epsilon).log()
    return -log_sum / len(targets)