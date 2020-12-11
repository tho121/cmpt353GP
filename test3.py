# weather_city.py - Jeff Wang - 301309384 - CMPT353
# cd mnt/c/users/17789/desktop/353/final

# python3 test3.py 

import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



if __name__ == '__main__':
    df_labelled = pd.read_csv('new2.csv')
    df_unlabelled = pd.read_csv('test2.csv')

    # print(df_labelled.iloc[:,0])
    X = df_labelled.drop(df_labelled.columns[0], axis=1)
    y = df_labelled.iloc[:,0]
    print(X, y)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = make_pipeline( 
        StandardScaler(), 
        RandomForestClassifier(n_estimators=400,
        max_depth=15, min_samples_leaf=15)
    )

    model.fit(X_train, y_train)
    print(model.score(X_valid, y_valid))

    X_perdiction = df_unlabelled.drop(df_unlabelled.columns[0], axis=1)
    print(X_perdiction)
    predictions = model.predict(X_perdiction)

    pd.Series(predictions).to_csv('res.csv', index=False, header=False)
