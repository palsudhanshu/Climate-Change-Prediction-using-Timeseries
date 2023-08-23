# Climate-Change-Prediction-using-Timeseries
This project uses time series analysis to predict future climate change. Time series analysis is a statistical method that studies the patterns and trends in data over time.
# Chapter 01: INTRODUCTION

# 1.1	Introduction
In this part, we will be gaining some insights into the topic-Time Series Analysis of Climate Change. The temperature at the Earth’s surface is counted as one of the most important environmental factors among all those factors that affect climate in a different way. Hence, modeling the variation of temperature and making dependable forecasts helps us to visualize the climatic conditions in a better manner. The temporal changes of the global surface temperature allow environmental researchers to communicate smoothly. Temperature affects all the other factors affecting the climate; therefore, it is very essential to study the temperature patterns and trends in order to align with other factors and trends in the environment. 
Generally, a structural time series data model decomposes into trend, seasonality, and residuals in trend components. A time series mathematically can be defined as 
xk = ak + bk + rk
where xk is the resulting time series, ak is a trend component, bk is a seasonal (periodic) component, and rk is a residual component that is often a stochastic time series signal.


# What is Climate Change?
Climate change, in general, refers to a long-term shift in temperature and weather patterns. The reasons for shifting might differ from a natural effect, human activities, or even the differences in the solar cycle [1].


# Why Time Series?
A time series is a set of repeated measurements of the same phenomenon, taken sequentially over time. 
Time series forecasting is a technique for the prediction of events through a sequence of time. It predicts future events by analyzing the trends of the past, on the assumption that future trends will hold similar to historical trends. 
Time Series analysis is a method of analyzing the data in order to extract more and more meaningful insights and statistics during a time period.

Time series analysis is useful for two major reasons:

1.	 It allows us to understand and compare things without losing the important, shared background of ‘time’.
2.	 It allows us to make forecasts [2].

The AR, MA, ARMA, and ARIMA models are used to forecast the observation at (t+1) using past data from earlier time spots. However, it is vital to ensure that the time series remains stationary over the observation period's historical data. If the time series is not stationary, we can use the differencing factor on the records to see if the time-series graph is stationary over time. [3]

1.	 AR i.e. Auto-Regressive Models: Auto Regression (AR) is a type of model that calculates the regression of past time series and estimates the current or future values in the series.
Yt = 1* y-1 + 2* yt-2 + 3 * yt-3 +............ + k * yt-k

2.	 MA i.e. Moving Average Models: This kind of model calculates the residuals or errors of past time series and calculates the present or future values in the series known as Moving Average (MA) model.
Yt = α₁* Ɛₜ-₁ + α₂ * Ɛₜ-₂ + α₃ * Ɛₜ-₃ + ………… + αₖ * Ɛₜ-ₖ

3.	 ARMA i.e Autoregressive moving average Models: The AR and MA models have been combined to create this model. For forecasting future values of the time series, this model considers the impact of previous lags as well as residuals. The coefficients of the AR model are represented by and the coefficients of the MA model are represented by.
Yt = 1* yt-1 + 1* t-1 + 2* yt-2 + 2 * t-2 + 3 * yt-3 + 3 * t-3 +............ + k * yt-k + k * t-k

All of these models provide some insight, or at least reasonably accurate predictions, over a particular time series. It also depends on which model is perfectly suited to your needs. If the probability of error in one model is low compared to other models, it is advisable to choose the model that gives the most accurate estimates. 

As shown in Figure 1, Kuwait is the country with the highest average temperature. And Kazakhstan is the country with the highest average temperature difference as shown in figure 2.

 
Figure 1: Countries with the Highest Average Temperature 

 
Figure 2: Countries with Highest Average Temperature Difference

# 1.2	Objective 
The main objectives of designing this project are mentioned below:
i. To study the trends followed by the temperature factors of climate
ii. To predict the future values for temperature using the seasonal ARIMA model.
iii. To find the most appropriate ARIMA model for our dataset in order to increase the efficiency of predicting the less erotic future values.

# 1.3	Motivation 
Climate impacts are already more widespread and severe than expected. We are locked into even worse impacts from climate change in the near term. Risks will escalate quickly with higher temperatures, often causing irreversible impacts of climate change.
We want to gain more and more knowledge about why and what affects the climate. Therefore, to achieve these goals, we want to investigate temperature changes and use these insights to predict future temperatures using ARIMA models. Also, we will examine the mathematical background of the various ARIMA models to see if we can make changes to the formula.

# 1.4	Language Used

The whole project is based on the Python language. Python is an interpreted, high-level, general-purpose programming language developed by Guido Van Rossum and initially released in 1991. ! We have imported a few libraries at different stages of our project. These Python libraries are very useful and made analysis and visualizing the data much easier.

NumPy:
NumPy is a Python library used to handle the array components. NumPy can be used to perform a wide variety of mathematical operations on arrays. 

Pandas: 
Pandas is the Python library that allows us to work with data in a very sequential manner. This package of Python is widely used for data analysis.

Matplotlib, Seaborn: 
Matplotlib and Seaborn are both packages that help us to visualize data in different forms. These are known as graphical plotting libraries of Python.

Statsmodels:
statsmodels is a Python module that provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests and statistical data exploration. 

pmdarima:
pmdarima is a Python package used to implement the ARIMA model and helps to identify the best model for your dataset while providing you the order of time series i.e. the values for p,d,q. It helps us to find the most accurate model with the least AIC value and return a fitted ARIMA model.

1.6	Deliverables/Outcomes
Designing the time series forecasting model and using the techniques of prediction in order to predict future values influenced by past values. We finally would have a model that will predict the future average temperature for the land for a few of the countries including India. The outcomes of the forecasting model shall be visualized and displayed.


Figure 3: Data Flow Chart
# Chapter 02: IMPLEMENTATION

# 3.1	Date Set Used
Berkeley’s Earth Surface Temperature (BEST) [4] dataset is used for this project. The dataset contains a total of 1.6 billion temperature values [5]. The dataset gets updated from time to time. The dataset has already eliminated repeated records. The dataset contains majorly 5 sub-datasets namely:
1.	GlobalTemperaturesByCity
2.	GlobalLandTemperaturesByCountry
3.	GlobalLandTemperaturesByMajorCity
4.	GlobalLandTemperaturesByState
5.	GlobalTemperatures
Each sub-dataset has its specified and segregated values. We have worked on two of them i.e. GlobalTemperatures and GlobalLandTemperaturesByCountry. We have visualized the global temperatures and then implemented our concerned ARIMA model on global temperatures by country.

3.2	Date Set Features
	3.2.1 Types of Data Set

The data set chosen is a time-varying dataset that has been recorded at each particular interval of the time period. Time-series data is a sequence of data points collected over time intervals, giving us the ability to track changes over time. Time-series data can track changes over milliseconds, days, or even years. [15] All time-series datasets have 3 things:
1.	The data that arrives is always recorded as a new entry to the database
2.	The data typically arrives in time order
3.	Time is a primary axis (i.e. time intervals can be either regular or irregular)
In other words, time-series data workloads are generally “append-only.” While they may need to
correct erroneous data after the fact, or handle delayed or out-of-order data, these are exceptions, not the norm.

		3.2.2 Number of Attributes, fields, and description of the data set

In GlobalTemperatures, there are nine attributes in total namely, ‘dt’ (describing the dates of the record in the format yyyy-mm-dd), ‘LandAverageTemperature’ (describing the average temperature of the globe on the corresponding date), ‘LandAverageTemperatureUncertainity’, ‘LandMaxTemperature’ (describing the maximum temperature on land), ‘LandMaxTemperatureUncertainity’, ‘LandMinTemperature’ (describing the minimum temperature on the land on the corresponding date), ‘LandMinTemperatureUncertainity’, ‘LandAndOceanAverageTemperature’ (describing the combined average land and ocean temperature), ‘LandAndOceanAverageTemperatureUncertainity’.
In GlobalLandTemperaturesByCountry, there are four attributes namely, ‘dt’ (describing the date of record), ‘AverageTemperatures’ (describing the average temperature on land in the corresponding country and date), ‘AverageTemperaturesUncertainity’, ‘Country’ (describing the location of record).
(NOTE: The uncertainty attributes are describing the statistical uncertainty calculation in the current averaging process intended to capture the portion of uncertainty introduced due to the noise and other factors that may prevent the basic data from being an accurate reflection of the climate at the measurement site) [16].


# 3.3 	Design of Problem Statement

The aim of this project is to predict future temperature values from the past recorded average temperature values and to analyze different meaningful insights from the time series dataset. In order to reach this goal, we will use ARIMA (the most used time series model to predict and analyze the land temperature of Berkeley’s Earth surface temperature dataset.


# 3.4 Algorithm / Pseudocode of the Project Problem

1.	Firstly, we need to do data cleaning. Our data was nearly clean; we just filled the NaN values with the last recorded values of temperature in the respective columns.   
2.	After that, we visualized our data on various parameters and found interesting insights about our dataset.
3.	We implemented ARIMA modeling to forecast the temperature. We used prima, a package of Python used to find the best ARIMA model for your dataset. 
4.	Fit the model
5.	Train the model with data values.
6.	Calculate the accuracy.
7.	And finally forecast your choice of Average temperature values.

# 3.5 Flow graph of the Minor Project Problem
 

Figure 4: Working of ARIMA Model

We know that in order to be able to apply different models, we first need to convert the series to a stationary time series. To achieve the same, apply a differential or integrated method that subtracts the t1 value from the t-value in the time series. If you still cannot get the stationary time series after applying the first derivative, apply the second derivative again.

The ARIMA model is very similar to the ARMA model, except that it contains another element known as Integrated (I). H. A derivative that represents I in the ARIMA model. That is, the ARIMA model is a combination of the set of differences already applied to the model to make it stationary, the number of previous lags, and the residual error to predict future values.

The parameters of the ARIMA model are defined as follows:
●	p: The number of lag observations included in the model, also called the lag order.
●	d: The number of times that the raw observations are differenced, also called the degree of differencing.
●	q: The size of the moving average window, also called the order of moving average. [6]
When adopting the ARIMA model over time, the underlying process that generated the observations is assumed to be the ARIMA process. This may seem obvious, but it helps motivate the need to confirm model assumptions with raw observations and residual errors in the model's predictions.

# 3.6   Screenshots of the various stages of the Project
1.	Let’s plot the specified dataset first.     
 
Figure 5: The plot of Average Temperatures And the uncertainty of all countries
















2.	Augmented Dickey-Fuller test for stationarity check on the specified dataset of a few countries. We can see that the data is stationary.

 
Figure 6: Stationarity check



















3.	The average temperature means yearly levels. We can see that in Cyprus, the average temperature is the highest and in the Gambia, the average temperature is the lowest.
  
Figure 7: The mean temperature levels



4.	Let’s visualize the variation in the average temperature of the countries.
 
Figure 8: Visualization of temperature variation




















5.	Visualization of average temperature for the years 2004-2013. We can observe that there is a rapid increase in average temperature in countries like India, the United States, Italy, Europe,  and Japan. We can also observe the huge difference in the average temperature of Russia in 2012 and 2013.
      Figure 9: Visualization of average temperature from 2004 to 2013

6.	Plotting of Average temperatures of India. 
 
Figure 10: India's average temperatures over the past years

7.	Augmented Dickey-Fuller test for stationarity check of India's average temperatures. We found a very weird observation that the data was stationary for all countries but not stationary for individual countries like India.
 
Figure 11: Stationarity check for India data

8.	Also checked stationarity using the Rolling statistics test, which is the visual test for stationarity. As we can see in the plot, the mean and variance is not constant and continuously varying at different instances of time, and hence the data is not stationary.
 
Figure 12: Rolling Statistics Test

9.	Moving average of India data set. We can clearly see the increasing trend in the dataset. And that the average temperature increased from 23.5º to 25.5º, that's 8.51% in over 100 years.
 
Figure 13: Yearly Average Temperature in India

10.	Choosing the best model for the dataset. We got our model ARIMA (5,1,2) (0,0,0) [0].

 
Figure 14: Choosing the best ARIMA model

11.	 Fitting the model. Now, we have the coefficients for the autoregressive equation.
 
Figure 15: Fitting model


 
# Chapter 3: RESULTS

4.1	Discussion on the Results Achieved
The forecasting model has successfully been able to predict the future values for the land temperature. We found that overall countries' data is stationary but when we analyzed the patterns of the country India, we found that the data was not stationary and hence needed to be made stationary before implementing the ARIMA model. 

 
Figure 16: Prediction Plot on Training Dataset
 

Figure 17: Prediction Plot on Testing Dataset

4.2	Application of the Minor Project
Our forecasting model predicts the correct values on the basis of past observed values and hence, can be used by weather researchers in order to predict and analyze the forecast.

4.3 	Limitation of the Minor Project
No model can be perfect after getting practically implemented in the first attempt. Hence, our model also has some limitations which are mentioned below:

1.	The model gives the testing RMSE error of 0.35, and hence there is a scope for accuracy in the model.
2.	The model takes a lot of time to train the data and then fit the best model to the dataset.

4.4	Future Work
We now have the predicted temperature values, and we look forward to analyzing the other factors affecting climate while keeping the temperature factor in mind. We also look forward to giving this model an appropriate interface to show its work more clearly.


# References
1.	https://www.un.org/en/climatechange/what-is-climate-change
2.	https://towardsdatascience.com/time-series-analysis-and-climate-change-7bb4371021e
3.	https://towardsdatascience.com/time-series-models-d9266f8ac7b0
4.	https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data
5.	https://redivis.com/datasets/1e0a-f4931vvyg
6.	https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
