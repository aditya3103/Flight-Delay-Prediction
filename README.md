# Flight-Delay-Prediction

# Installation

To install the project, follow these steps:

1. Clone the repository to your local machine using the command: `git clone <https://github.com/aditya3103/Flight-Delay-Prediction>`
2. Install all dependencies by running: `pip install skfuzz` `pip install numpy` `pip install beautifulsoup4`
3. Run the python files in the main branch.

# Name of the Search Algorithm

BFS and A-Star Search

# Approach/Idea

Breadth First Search is a type of level order traversal which helps find the minimum distance but fails when edge costs are not same. A-Star Search is a more informed search where it makes use of a heuristic function which always ensures that the algorithm converges to the goal node of the search tree.

# Explanation

In both the search algorithms, there is a slight variation. At every node, we make use of an online API to get the weather data of that particlar node (airport) and feed that data into a fuzzy logic which gives a float value (multiplier). This multiplier helps in judging the delay in traveling the edge (edge cost is the average flight travel time between the nodes). Finally, after reaching the goal node, the average flight time is subtracted from the delays accumulated over the course of the algorithm, this is outputted as the flight delay in minutes using real time data.

# Visuals and GUI

The project includes a graphical user interface (GUI) that allows users to input a graph and run the BFS algorithm and the A-Star Search Algorithm. The GUI displays the delay in minutes and the terminal shows the sequence of nodes visited by the algorithm to arrive at the goal node (which is approximately same as the actual flight path assuming the nodes are densely situated geographically)



# Conclusion

In conclusion, BFS fails to give a good estimate of the delay because obvious shortcomings in the algorithm to calculate the shortest path when the edges are non uniform. A-Star Search does a very fine job in estimating the delay as it gives very realisitic values, close to what we may encounter in daily lives.

# Name of Search Algorithms

Genetic, Hill Climbing and Particle Swarm Optimization.

# Approach/Idea

Our project is a flight delay predictor that uses machine learning algorithms to analyze historical flight data and predict the likelihood of delays for future flights. Our approach involves implementing a Random Forest algorithm, along with a variety of hyperparameter optimization techniques such as Genetic Algorithm, PSO, and Hill Climbing to maximize the accuracy of our predictions. We also train our models on both the raw and normalized datasets to evaluate the impact of data normalization on performance.

# Explanation

Our flight delay predictor uses a dataset that contains flight information, including flight time, origin and destination airports, airlines, distance between airports, and scheduled and actual elapsed times. We use the Random Forest algorithm, which is a type of decision tree-based ensemble algorithm that combines multiple decision trees to generate a final prediction. To optimize our algorithm, we implement hyperparameter optimization techniques to tune the parameters of our model and increase its accuracy. We also pre-process our data by normalizing it to ensure that our models have equalized distributions and reduce the impact of outliers on the training process.

We evaluate the performance of our models using several different metrics, including mean absolute error (MAE), mean squared error (MSE), root mean squared error (RMSE), and R2 score. To visualize our results, we create bar graphs that compare the performance of each model across these metrics.

# Visualisations and Variations

The following bar graphs show the comparison of each model based on MAE, MSE, RMSE, and R2 score:

Bar Graph for  **MAE**

![image](https://user-images.githubusercontent.com/87059885/233945104-8c2aac47-846b-425e-a8d5-5ac9009919da.png)





Bar Graph for **MSE**

![image](https://user-images.githubusercontent.com/87059885/233943652-7325b583-7fde-4985-b079-15ac6ab6584b.png)


Bar Graph for R2 Score

![image](https://user-images.githubusercontent.com/87059885/233943736-3e649a05-8739-4d03-bc3a-ec95ccc5ad48.png)


Bar Graph for RMSE

![image](https://user-images.githubusercontent.com/87059885/233943854-a5d5bbc5-80a0-42a2-a1e6-d883922df8b8.png)


# Conclusion

Our results indicate that our Random Forest model, trained using hyperparameter optimization with the Genetic Algorithm, outperforms the other models we tested, with a lower MAE, MSE, and RMSE, and a higher R2 score. Additionally, we found that normalizing our data had a positive impact on the performance of our models, improving their accuracy in some cases.

Future work on this project could involve expanding the scope of our predictor to include additional factors that may impact flight delays, such as weather conditions or air traffic control. We could also explore different algorithms and optimization techniques to improve the performance of our models even further.
