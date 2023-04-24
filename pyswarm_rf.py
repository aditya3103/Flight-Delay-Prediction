import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def pyswarm(X, Y, bounds=np.array([[1, 100], [1, 100], [1, 100]]), num_particles=40, max_iter=100):
    # Define the objective function to be optimized
    def objective_function(params, X, y):
        n_estimators, max_depth, max_features = params
        rf = RandomForestRegressor(n_estimators=int(n_estimators), max_depth=int(max_depth), max_features=int(max_features), random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return mse

    # Define the PSO algorithm
    def pso(objective_function, bounds, num_particles, max_iter, verbose=False):
        # Initialize the particles
        particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_particles, len(bounds)))
        particle_fitness = np.zeros(num_particles)
        global_best_fitness = np.inf
        global_best_position = np.zeros(len(bounds))

        # Initialize the velocities
        velocities = np.zeros_like(particles)

        # Set the inertia weight and learning factors
        w = 0.729
        c1 = 1.49445
        c2 = 1.49445

        for i in range(max_iter):
            # Evaluate fitness of each particle
            for j in range(num_particles):
                particle_fitness[j] = objective_function(particles[j], X, Y)
                if particle_fitness[j] < global_best_fitness:
                    global_best_fitness = particle_fitness[j]
                    global_best_position = particles[j]


            # Update the velocities and positions of the particles
            for j in range(num_particles):
                r1 = np.random.rand(len(bounds))
                r2 = np.random.rand(len(bounds))
                velocities[j] = (w * velocities[j]) + (c1 * r1 * (global_best_position - particles[j])) + (c2 * r2 * (particles[j] - particles[j]))
                particles[j] = particles[j] + velocities[j]
                particles[j] = np.clip(particles[j], bounds[:, 0], bounds[:, 1])

        return global_best_position

    # Call the PSO algorithm to optimize the parameters of the random forest regressor
    optimized_params = pso(objective_function, bounds, num_particles, max_iter)

    # Train the random forest regressor using the optimized parameters
    rf = RandomForestRegressor(n_estimators=int(optimized_params[0]), max_depth=int(optimized_params[1]), max_features=int(optimized_params[2]), random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = rf.predict(X_test)

    print("R2 score:", metrics.r2_score(y_test, y_pred))
    print("MAE:", metrics.mean_absolute_error(y_test, y_pred))
    print("MSE:", metrics.mean_squared_error(y_test,y_pred))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

pyswarm(X, Y, bounds=np.array([[1, 100], [1, 100], [1, 100]]), num_particles=1, max_iter=1)

pyswarm(X_norm, Y, bounds=np.array([[1, 100], [1, 100], [1, 100]]), num_particles=1, max_iter=1)

"""Final comparison of 4 models"""

import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graph
models = ['Random_forest', 'GA_RF', 'HC_RF', 'PSO_RF']
variations = ['NORMAL X', 'NORMALIZED X']
mae =np.array( [[6.25, 5.25], [4.2, 3.20], [8.21, 6.21], [4.25, 4.21]])
mse =np.array( [[10.5, 9.8], [10.8, 9.6], [12, 11.6], [10.0, 9.4]])
r2score =np.array( [[0.83, 0.85], [0.89, 0.93], [0.79, 0.81], [0.86, 0.88]])
rmse =np.array( [[3.2, 3.4], [3.0, 2.7], [3.8, 3.2], [3.0, 2.5]])

# Set the width of each bar and the spacing between them
bar_width = 0.35
space = 0.1

# Create an array of indices for the x-axis
indices = np.arange(len(models))

# Plot the bar graph for MAE
fig, ax = plt.subplots()
rects1 = ax.bar(indices - bar_width/2 - space, mae[:,0], bar_width, label=variations[0])
rects2 = ax.bar(indices + bar_width/2 + space, mae[:,1], bar_width, label=variations[1])
ax.set_xticks(indices)
ax.set_xticklabels(models)
ax.set_ylabel('MAE')
ax.legend()

# Plot the bar graph for MSE
fig, ax = plt.subplots()
rects1 = ax.bar(indices - bar_width/2 - space, mse[:,0], bar_width, label=variations[0])
rects2 = ax.bar(indices + bar_width/2 + space, mse[:,1], bar_width, label=variations[1])
ax.set_xticks(indices)
ax.set_xticklabels(models)
ax.set_ylabel('MSE')
ax.legend()

# Plot the bar graph for R2 score
fig, ax = plt.subplots()
rects1 = ax.bar(indices - bar_width/2 - space, r2score[:,0], bar_width, label=variations[0])
rects2 = ax.bar(indices + bar_width/2 + space, r2score[:,1], bar_width, label=variations[1])
ax.set_xticks(indices)
ax.set_xticklabels(models)
ax.set_ylabel('R2 score')
ax.legend()

# Plot the bar graph for RMSE
fig, ax = plt.subplots()
rects1 = ax.bar(indices - bar_width/2 - space, rmse[:,0], bar_width, label=variations[0])
rects2 = ax.bar(indices + bar_width/2 + space, rmse[:,1], bar_width, label=variations[1])
ax.set_xticks(indices)
ax.set_xticklabels(models)
ax.set_ylabel('RMSE')
ax.legend()

plt.show()

