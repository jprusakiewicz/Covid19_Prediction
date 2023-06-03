from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, x, y) -> dict:
    y_pred = model.predict(x)
    # y_pred_proba = model.predict_proba(x)

    r2 = r2_score(y, y_pred)
    rmse = mean_squared_error(y, y_pred,
                              squared=False)  # squared: If True returns MSE value, if False returns RMSE value. (we want RMSE)
    return {"RMSE": rmse, "r2": r2}