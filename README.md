# Supervised-Machine-Learning-Model-to-predict-Computer-Performance

## Description of program

- a Python Supervised Machine Learning Linear Regression model to train and predict relative computer performance using independent hardware attributes from a UCI repository dataset.
- Extended the model by having a Tkinter GUI to make it more user friendly.
- The program can plot a scatter graph of a chosen attribute inrelation to relative computer performance.
- This model is supervised as the algorithm learns on a labeled dataset. The type of supervised learning is linear regression where the algorithm finds a line (y=mx+c) that best fits for scatter of data points on the plot. If there is a strong correlation, the line can be used to predict ouputs from given inputs. In my program, I split the dataset into test and training data set, where 90 percent is used for the training dataset and 10% is used for the test dataset. A high percentage is used for the training dataset to increase accuracy of the prediction.

## Installation
- need to install the libraries: numpy, pandas, sklearn, matplotlib, pickle, tkinter
- command on cmd: `pip install `[name of library]

## What I have learned
- Most significant attribute which positively affected computer performance were MMAX (maximum main memory)
- How linear regression is used computationally
- Libraries: pandas, sklearn, pickle, matplotlib and how each of them played a pivital role in gathering, training, loading and plotting data from the dataset.
  (More specific) Used the Python libraries ‘Pandas’ and ‘Numpy’ to read key attributes from dataset, ‘SKLearn’ and ‘Pickle’ to train & load the model. 
- Enabled visualisation of relationship between an attribute in the dataset and the relative performance using matplotlib.

## How I could improve
- [x] Extend the model by creating a user friendly deploy to streamlit 
- [ ] Boost the accuracy of the model by using datasets with more instances to get more accurate the results. Treat all missing values. Detect and remove all outliers.

## Ideas for the future
- Supervised model that uses classification or clustering 
- Store data based on program analysing drawn pictures e.g. to predict if people have parkinsons or dementia
- Gather data yourself from an experiment and use them on the model
- Unsupervised ML model

## Resources used to help
- 'Tech with Tim Machine Learning Tutorial [#1-4]' on Youtube and also on his website. 
  Youtube:https://www.youtube.com/channel/UC4JX40jDee_tINbkjycV4Sg
  Website: https://techwithtim.net/tutorials/machine-learning-python/introduction/
  Tim's tutorial focused on using a student performance database to predict student's final grades. 
  To develop a stronger understanding I used the Computer Hardware database (from UCI repository database).
  Additionally I am interested to investigate the relationship between certain hardware attributes to the relative computer performance. 
  
- The dataset I used is from UCI Machine Learning repository: https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
  Accessed in 2020
  Creator:
  Phillip Ein-Dor and Jacob Feldmesser
  Ein-Dor: Faculty of Management
  Tel Aviv University; Ramat-Aviv;
  Tel Aviv, 69978; Israel
