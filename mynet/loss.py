import math
from utils import debatch


class MSELoss:
    pass



@debatch
def CrossEntropyLoss(sample):
    y_target, y_pred = sample
    if len(y_target) != len(y_pred):
        raise ValueError(
            "Target vector length (%s) has to be same with "
            "predicted distribution vector length (%s)"
            % (len(y_target), len(y_pred))
        )
    if not math.isclose(sum(y_pred), 1):
        raise ValueError(
            "Predicted probability distribution has to sum up to 1 (100%)"
        )
    if sum(y_target) != 1:
        raise ValueError(
            "Target vector values have to sum up to 1"
        )
    y = y_target.index(max(y_target))
    return -math.log(y_pred[y] or 1e-100)
