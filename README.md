# Restaurant Revenue Prediction

Link to competition: https://www.kaggle.com/c/restaurant-revenue-prediction

The task is to create a model from training data set and predict revenue for test dataset.

The approach I have used is to convert given open date to days and then perform perform necessary preprocessing to ensure that regressor can take it as input.

I have used `RandomForestRegressor` ensemble with 200 estimators to perform regression. 
