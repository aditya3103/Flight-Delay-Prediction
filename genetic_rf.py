def feature_selection_with_ga(X, Y, population_size=1, num_generations=1, mutation_rate=0.1):
    # define the fitness function for the genetic algorithm
    def fitness_function(solution, X, Y):
        # get the indices of the selected features
        feature_indices = [i for i in range(len(solution)) if solution[i] == 1]
        # select the corresponding features from the dataset
        X_selected = X.iloc[:, feature_indices]

        # split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.2, random_state=0)

        # train a random forest regressor on the training set
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, Y_train)

        # make predictions on the testing set
        Y_pred = model.predict(X_test)

        # compute the mean absolute error of the predictions
        mae = mean_absolute_error(Y_test, Y_pred)

        # return the inverse of the mean absolute error as the fitness value
        return 1 / mae

    # initialize the population
    population = np.random.randint(2, size=(population_size, X.shape[1]))

    # evolve the population for the specified number of generations
    for generation in range(num_generations):
        # evaluate the fitness of each solution in the population
        fitness_scores = [fitness_function(solution, X, Y) for solution in population]

        # select the parents for the next generation
        parent_indices = np.random.choice(population_size, size=population_size, p=fitness_scores/np.sum(fitness_scores), replace=True)
        parents = population[parent_indices]

        # create the next generation by applying crossover and mutation operators to the parents
        offspring = []
        for i in range(population_size):
            parent1 = parents[np.random.randint(len(parents))]
            parent2 = parents[np.random.randint(len(parents))]
            child = parent1.copy()
            for j in range(len(child)):
                if np.random.rand() < mutation_rate:
                    child[j] = 1 - child[j]
            offspring.append(child)
        population = np.array(offspring)

    # select the best solution from the final population
    fitness_scores = [fitness_function(solution, X, Y) for solution in population]
    best_solution = population[np.argmax(fitness_scores)]

    # print the indices of the selected features
    selected_features = [i for i in range(len(best_solution)) if best_solution[i] == 1]
    print("Selected features:", selected_features)

    # select the corresponding features from the dataset
    X_selected = X.iloc[:, selected_features]

    # split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.2, random_state=0)

    # train a random forest regressor on the training set
    model = RandomForestRegressor(n_estimators=1000, random_state=0)
    model.fit(X_train, Y_train)

    # make predictions on the testing set
    Y_pred = model.predict(X_test)
    # Print evaluation metrics
    print("R2 score:", metrics.r2_score(y_test, y_pred))
    print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

feature_selection_with_ga(X, Y, population_size=4, num_generations=2, mutation_rate=0.1)

feature_selection_with_ga(X_norm, Y, population_size=4, num_generations=2, mutation_rate=0.1)
