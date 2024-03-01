import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier


def rolling_averages(group, cols, newCols):

    group = group.sort_values("date")
    rollingStats = group[cols].rolling(3, closed="left").mean()
    group[newCols] = rollingStats
    group = group.dropna(subset=newCols) # Removing missing values that can occur if there aren't previous 3 games

    return group

def make_predictions(data, predictors):

    train = matches[matches["date"] < "2022-01-01"]
    test = matches[matches["date"] >= "2022-01-01"]

    rf.fit(train[predictors], train["target"])

    predictions = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], prediction=predictions, index=test.index))
    accuracy = accuracy_score(test["target"],predictions)
    precision = precision_score(test["target"], predictions)

    return combined, accuracy, precision



if __name__ == "__main__":

    #Loading the data
    matches = pd.read_csv("/Users/onur/Python-VSCode/Match Winner Prediction/matches.csv", index_col=0)

    #Handling the data
    matches["date"] = pd.to_datetime(matches["date"])
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes
    matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
    matches["day_code"] = matches["date"].dt.dayofweek
    matches["target"] = (matches["result"] == "W").astype("int")

    #Splitting the dataset
    train = matches[matches["date"] < "2022-01-01"]
    test = matches[matches["date"] >= "2022-01-01"]

    #Loading the RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

    #Setting the Predictors
    predictors = ["venue_code", "opp_code", "hour", "day_code"]

    #Training the Model
    rf.fit(train[predictors], train["target"])


    predictions = rf.predict(test[predictors])
    acc = accuracy_score(test["target"],predictions)
    precision = precision_score(test["target"], predictions)
    print(f"Accuracy: {acc}, Precision: {precision}")

    combined = pd.DataFrame(dict(actual=test["target"], prediction=predictions))
    pd.crosstab(index=combined["actual"], columns=combined["prediction"])


    #Improving the performance of the model
    cols = ["gf","ga","sh","sot","dist","fk","pk","pkatt"]
    newCols = [f"{col}_rolling" for col in cols]
    matchesRolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, newCols))
    matchesRolling = matchesRolling.droplevel("team")
    matchesRolling.index = range(matchesRolling.shape[0])

    #Training model with new dataset
    combined, accuracy, precision = make_predictions(matchesRolling, predictors+newCols)
    print(f"Accuracy: {accuracy}, Precision: {precision}")



