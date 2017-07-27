import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="desired mode", type = int)
args = parser.parse_args()

sns.set(style="whitegrid", palette="muted")
current_palette = sns.color_palette()
if args.mode==1:
    df = pd.read_csv("./data/2017.csv")
    df.head(1)
    corrolmatrix = df.corr()
    sns.heatmap(corrolmatrix, vmax=1, square=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=40)
    plt.show()

if args.mode==2:
    df = pd.read_csv("./data/2017.csv")
    df.head(1)
    originalResults = df['Happiness.Score']
    dataWithNoResults = df.drop(['Country','Happiness.Score', 'Happiness.Rank','Whisker.high','Whisker.low'], axis=1)
    dataWithNoResults_train, dataWithNoResults_test, originalResults_train, originalResults_test = train_test_split(dataWithNoResults, originalResults, test_size=0.6, random_state=42)
    lm = LinearRegression()
    lm.fit(dataWithNoResults_train,originalResults_train)
    print('Coefficients: \n', lm.coef_)
    predict = lm.predict(dataWithNoResults_test)
    plt.scatter(originalResults_test,predict)
    plt.xlabel("Real Happiness Score")
    plt.ylabel("Predicted Happiness Score")
    plt.show()
