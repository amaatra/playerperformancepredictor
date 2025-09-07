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
filtered_df1 = None
filtered_df2 = None

mode = input("Type 'Player' to compare two players, 'Club' to compare two clubs, or 'Fantasy' to create and compare your own teams")
mode = str(mode.title())

if ((mode == "Player") or (mode == "Club")):
    team_list = players_df[mode].tolist()
    teamname1 = input("First " + mode + ": ")

    # Getting the two clubs/players

    if(mode == "Club"):
        teamname1 = teamname1.upper()
    while not bad_input_teamname(teamname1, team_list):
        teamname1 = input("Please try again. First " + mode + ": ")

    teamname2 = input("Second " + mode + ": ")
    if(mode == "Club"):
        teamname1 = teamname1.upper()
    while not bad_input_teamname(teamname2, team_list):
        teamname2 = input("Please try again. First " + mode + ": ")

    # Filtering the dataframe based on the input

    filtered_df1 = (players_df[players_df[mode] == teamname1].groupby("Year", as_index = False)["G"].sum())
    filtered_df2 = (players_df[players_df[mode] == teamname2].groupby("Year", as_index = False)["G"].sum())


elif (mode=="Fantasy"):
    # Two lists to represent the two fantasy teams

    fantasy_team1 = [None, None, None, None, None, None, None, None, None, None, None]
    fantasy_team2 = [None, None, None, None, None, None, None, None, None, None, None]

    # Adding players to the list

    for i in range(11):
        fantasy_team1[i] = input("Player " + str(i+1) + ": ")
    print("First Team: " + str(fantasy_team1))

    for i in range(11):
        fantasy_team2[i] = input("Player " + str(i+1) + ": ")
    print("Second Team: " + str(fantasy_team2))

    # Filtering out rows of the dataframe

    fantasyteam_df1 = players_df.loc[players_df['Player'].isin(fantasy_team1)]
    fantasyteam_df2 = players_df.loc[players_df['Player'].isin(fantasy_team1)]

    filtered_df1 = (fantasyteam_df1.groupby("Year", as_index = False)["G"].sum())
    filtered_df2 = (fantasyteam_df2.groupby("Year", as_index = False)["G"].sum())

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