# Flight-Delay-Prediction
**Name of Search Algorithms used:** Genetic, Hill Climbing and Particle Swarm Optimization.

**Approach/Idea:** Our project is a flight delay predictor that uses machine learning algorithms to analyze historical flight data and predict the likelihood of delays for future flights. Our approach involves implementing a Random Forest algorithm, along with a variety of hyperparameter optimization techniques such as Genetic Algorithm, PSO, and Hill Climbing to maximize the accuracy of our predictions. We also train our models on both the raw and normalized datasets to evaluate the impact of data normalization on performance.

**Explanation:** Our flight delay predictor uses a dataset that contains flight information, including flight time, origin and destination airports, airlines, distance between airports, and scheduled and actual elapsed times. We use the Random Forest algorithm, which is a type of decision tree-based ensemble algorithm that combines multiple decision trees to generate a final prediction. To optimize our algorithm, we implement hyperparameter optimization techniques to tune the parameters of our model and increase its accuracy. We also pre-process our data by normalizing it to ensure that our models have equalized distributions and reduce the impact of outliers on the training process.

We evaluate the performance of our models using several different metrics, including mean absolute error (MAE), mean squared error (MSE), root mean squared error (RMSE), and R2 score. To visualize our results, we create bar graphs that compare the performance of each model across these metrics.

**Visualisations:** The following bar graphs show the comparison of each model based on MAE, MSE, RMSE, and R2 score:

Bar Graph for  **MAE**

![image](https://user-images.githubusercontent.com/87059885/233944902-255ac29d-f11f-4011-b285-70084ae63925.png)





Bar Graph for **MSE**

![image](https://user-images.githubusercontent.com/87059885/233943652-7325b583-7fde-4985-b079-15ac6ab6584b.png)


Bar Graph for R2 Score

![image](https://user-images.githubusercontent.com/87059885/233943736-3e649a05-8739-4d03-bc3a-ec95ccc5ad48.png)


Bar Graph for RMSE

![image](https://user-images.githubusercontent.com/87059885/233943854-a5d5bbc5-80a0-42a2-a1e6-d883922df8b8.png)


**Conclusion:** Our results indicate that our Random Forest model, trained using hyperparameter optimization with the Genetic Algorithm, outperforms the other models we tested, with a lower MAE, MSE, and RMSE, and a higher R2 score. Additionally, we found that normalizing our data had a positive impact on the performance of our models, improving their accuracy in some cases.

Future work on this project could involve expanding the scope of our predictor to include additional factors that may impact flight delays, such as weather conditions or air traffic control. We could also explore different algorithms and optimization techniques to improve the performance of our models even further.
