import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from difflib import get_close_matches

# Functions

def show():
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

def predict(year):
    pred1 = np.exp(model1.predict([[year]])[0])
    pred2 = np.exp(model2.predict([[year]])[0])

    if (pred1 > pred2):
        return teamname1 + " is more likely to win. "
    elif (pred1 < pred2):
        return teamname2 + " is more likely to win. "
    else:
        return "It's a tie."

def bad_input():
    print("Invalid input.")
    quit()

def bad_input_teamname(user_input, teamlist):
    if user_input not in teamlist:
        print(user_input + " not found in list. Please try again.")
        return False
    return True

# Getting the dataframe

file_path = "all_players.csv"
players_df = pd.read_csv(file_path)

player_or_team = input("Would you like to select two teams or two players? Type 'Player' or 'Club': ")
player_or_team = str(player_or_team.title())

team_list = players_df[player_or_team].tolist()

teamname1 = input("First " + player_or_team + ": ")
if(player_or_team == "Club"):
    teamname1 = teamname1.upper()
while not bad_input_teamname(teamname1, team_list):
    teamname1 = input("Please try again. First " + player_or_team + ": ")

teamname2 = input("Second " + player_or_team + ": ")
if(player_or_team == "Club"):
    teamname1 = teamname1.upper()
while not bad_input_teamname(teamname2, team_list):
    teamname2 = input("Please try again. First " + player_or_team + ": ")

bad_input_teamname(teamname2, team_list)


# Filtering dataframes

filtered_df1 = (
    players_df[players_df[player_or_team] == teamname1]
    .groupby("Year", as_index = False)["G"]
    .sum()
    )
filtered_df2 = (
    players_df[players_df[player_or_team] == teamname2]
    .groupby("Year", as_index = False)["G"]
    .sum()
    )

# Setup of dataframes and regression models

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

option = input("Type 'show' to show the graph of both teams/players or 'predict' to see which team/player is more likely to win: ")
option = option.lower()

if (option == "show"):
    show()
elif (option == "predict"):
    year = int(input("Enter the year (as an integer) you wish to predict the match for: "))
    print(str(predict(year)))

    if((r2_team1 < 0.5) | (r2_team2 < 0.5)):
        print("This prediction may not be accurate!")
    
    see_r2 = input("See R^2 values? (y/n)")
    if(see_r2 == "y"):
        print("R^2 value for " + teamname1 + ": " + str(r2_team1))
        print("R^2 value for " + teamname2 + ": " + str(r2_team2))
else:
    bad_input()