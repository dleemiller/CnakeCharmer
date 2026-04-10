import math


def model_error_assess(n):
    """Assess prediction error of a simple model on deterministic data.

    Generates predictions and targets, computes MSE, MAE, and max error.
    Returns (mse, mae, max_error).
    """
    mse = 0.0
    mae = 0.0
    max_err = 0.0

    for i in range(n):
        target = math.sin(i * 0.1) * 10.0
        pred = math.sin(i * 0.1 + 0.05) * 9.8
        err = target - pred
        mse += err * err
        ae = abs(err)
        mae += ae
        if ae > max_err:
            max_err = ae

    mse /= n if n > 0 else 1
    mae /= n if n > 0 else 1
    return (round(mse, 6), round(mae, 6), round(max_err, 6))
