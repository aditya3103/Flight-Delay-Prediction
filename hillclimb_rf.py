def hill_climb(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    max_depth_range = range(1, 10)
    min_samples_split_range = range(2, 10)
    max_features_range = range(1, len(X.columns))

    def rf_score(params):
        max_depth, min_samples_split, max_features = params
        rf = RandomForestRegressor(max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        return metrics.r2_score(y_test, y_pred)

    def hill_climb(func, ranges, max_iter=1, step=2):
        best_params = None
        best_score = float('-inf')
        params = [choice(range) for range in ranges]
        for i in range(max_iter):
            scores = []
            for j, diff in product(range(len(params)), [-step, step]):
                new_params = params.copy()
                new_params[j] += diff
                if new_params[j] not in ranges[j]:
                    continue
                score = func(new_params)
                scores.append((score, new_params))
                if score > best_score:
                    best_score = score
                    best_params = new_params
            if not scores:
                break
            scores.sort(reverse=True)
            params = scores[0][1]
        return best_params

    ranges = [max_depth_range, min_samples_split_range, max_features_range]
    best_params = hill_climb(rf_score, ranges)

    rf = RandomForestRegressor(max_depth=best_params[0], min_samples_split=best_params[1], max_features=best_params[2], random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("R2 score:", metrics.r2_score(y_test, y_pred))
    print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
    print("MSE:", metrics.mean_squared_error(y_test,y_pred))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

hill_climb(X, Y)

hill_climb(X_norm, Y)
