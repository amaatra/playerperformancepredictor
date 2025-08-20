import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

file_path = "all_players.csv"
players_df = pd.read_csv(file_path)

teamname1 = input("First Team: ")
teamname2 = input("Second Team: ")

option = input("What would you like to do? Type 'show' to show the graph or 'predict' to see which team is likely to win")

filtered_df1 = (
    players_df[players_df['Club'] == teamname1]
    .groupby("Year", as_index = False)["G"]
    .mean()
)
filtered_df2 = (
    players_df[players_df['Club'] == teamname2]
    .groupby("Year", as_index = False)["G"]
    .mean()
)


X1 = filtered_df1["Year"].values.reshape(-1, 1)
y1 = filtered_df1["G"]

X1_log = X1[y1 > 0]
y1_log = np.log(y1[y1 > 0])

model1 = LinearRegression().fit(X1_log, y1_log)
trendline1 = np.exp(model1.predict(X1))
r2_team1 = model1.score(X1_log, y1_log)

X2 = filtered_df2["Year"].values.reshape(-1, 1)
y2 = filtered_df2["G"]

X2_log = X2[y2 > 0]
y2_log = np.log(y2[y2 > 0])

model2 = LinearRegression().fit(X2_log, y2_log)
trendline2 = np.exp(model2.predict(X2))
r2_team2 = model2.score(X2_log, y2_log)

if (option == "show"):
    plt.plot(X1, trendline1, color="blue")
    plt.plot(X2, trendline2, color="orange")

    plt.plot(X1, y1, label=teamname1, marker="o")
    plt.plot(X2, y2, label=teamname2, marker="s")

    plt.title("Goals over time")
    plt.xlabel("Year")
    plt.ylabel("Goals")
    plt.legend()
    plt.grid(True)


    plt.show()
elif (option == "predict"):
    year = int(input("Which year? Please enter a number only"))
    pred1 = np.exp(model1.predict([[year]])[0])
    pred2 = np.exp(model2.predict([[year]])[0])

    if (pred1 > pred2):
        print(teamname1 + " is more likely to win. ")
    elif (pred1 < pred2):
        print(teamname2 + " is more likely to win. ")
    else:
        print("It's a tie.")

print("R^2 value for " + teamname1 + ": " + str(r2_team1))
print("R^2 value for " + teamname2 + ": " + str(r2_team2))