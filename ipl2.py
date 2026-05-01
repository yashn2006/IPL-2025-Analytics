import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

# ----------------------------
# 1️⃣ Load Data
# ----------------------------
batsmen = pd.read_csv("IPL2025Batters.csv")
bowlers = pd.read_csv("IPL2025Bowlers.csv")

# ----------------------------
# 2️⃣ Fill Null Values in Batsmen
# ----------------------------
batsmen['Runs'] = batsmen['Runs'].fillna(batsmen['Runs'].mean())
batsmen['SR'] = batsmen['SR'].fillna(batsmen['SR'].mean())
batsmen['100s'] = batsmen['100s'].fillna(batsmen['100s'].median())
batsmen['50s'] = batsmen['50s'].fillna(batsmen['50s'].median())
batsmen['4s'] = batsmen['4s'].fillna(batsmen['4s'].mode()[0])
batsmen['6s'] = batsmen['6s'].fillna(batsmen['6s'].mode()[0])

# ----------------------------
# 3️⃣ Fill Null Values in Bowlers
# ----------------------------
bowlers['WKT'] = bowlers['WKT'].fillna(bowlers['WKT'].mean())
bowlers['MAT'] = bowlers['MAT'].fillna(bowlers['MAT'].median())
bowlers['INN'] = bowlers['INN'].fillna(bowlers['INN'].median())
bowlers['RUNS'] = bowlers['RUNS'].fillna(bowlers['RUNS'].mean())
bowlers['ECO'] = bowlers['ECO'].fillna(bowlers['ECO'].median())
bowlers['SR'] = bowlers['SR'].fillna(bowlers['SR'].mode()[0])

# ----------------------------
# 4️⃣ Rename Columns
# ----------------------------
batsmen = batsmen.rename(columns={
    "Player Name": "Player",
    "Team": "Team",
    "Runs": "Runs",
    "Matches": "Matches",
    "Inn": "Innings_Bat",
    "No": "NotOuts",
    "HS": "HighScore",
    "AVG": "Batting_Avg_Provided",
    "BF": "Balls_Faced",
    "SR": "StrikeRate_Provided",
    "100s": "Hundreds",
    "50s": "Fifties",
    "4s": "Fours",
    "6s": "Sixes"
})

bowlers = bowlers.rename(columns={
    "Player Name": "Player",
    "Team": "Team",
    "WKT": "Wickets",
    "MAT": "Matches",
    "INN": "Innings_Bowl",
    "OVR": "Overs",
    "RUNS": "Runs_Conceded",
    "BBI": "Best_Bowling",
    "AVG": "Bowling_Avg_Provided",
    "ECO": "Economy_Provided",
    "SR": "StrikeRate_Bowling",
    "4W": "FourW",
    "5W": "FiveW"
})

# ----------------------------
# 5️⃣ Convert Overs to Balls
# ----------------------------
bowlers["Balls_Bowled"] = (bowlers["Overs"].astype(float) * 6).round()

# ----------------------------
# 6️⃣ Merge Datasets
# ----------------------------
df = pd.merge(batsmen, bowlers, on=["Player", "Team", "Matches"], how="outer").fillna(0)

# ----------------------------
# 7️⃣ Batting Analysis
# ----------------------------
df["Batting_Avg"] = np.where(df["Matches"] > 0, df["Runs"] / df["Matches"], 0)
df["Strike_Rate"] = np.where(df["Balls_Faced"] > 0, (df["Runs"] / df["Balls_Faced"]) * 100, 0)

print("\nTop Run Scorers:")
print(df[["Player", "Runs", "Batting_Avg", "Strike_Rate"]].sort_values(by="Runs", ascending=False).head())

# ----------------------------
# 8️⃣ Bowling Analysis
# ----------------------------
df["Bowling_Avg"] = np.where(df["Wickets"] > 0, df["Runs_Conceded"] / df["Wickets"], 0)
df["Economy"] = np.where(df["Balls_Bowled"] > 0, (df["Runs_Conceded"] / df["Balls_Bowled"]) * 6, 0)

print("\nTop Wicket Takers:")
print(df[["Player", "Wickets", "Bowling_Avg", "Economy"]].sort_values(by="Wickets", ascending=False).head())

# ----------------------------
# 9️⃣ Overall Insights
# ----------------------------
print("\nOverall Insights:")
print("Total Runs Scored:", df["Runs"].sum())
print("Total Wickets Taken:", df["Wickets"].sum())
print("Average Runs per Player:", np.mean(df["Runs"]))

# ----------------------------
# 🔟 Special Awards
# ----------------------------
orangecap = df.iloc[df["Runs"].idxmax()][["Player", "Runs"]]
purplecap = df.iloc[df["Wickets"].idxmax()][["Player", "Wickets"]]
most_centuries = df.iloc[df["Hundreds"].idxmax()][["Player", "Hundreds"]]

print("\nSpecial Awards:")
print("Orange Cap (Most Runs):", orangecap.to_dict())
print("Purple Cap (Most Wickets):", purplecap.to_dict())
print("Most Centuries:", most_centuries.to_dict())

# ----------------------------
# 1️⃣1️⃣ Top Sixes and Fours
# ----------------------------
top_sixes = df[["Player", "Sixes"]].sort_values(by="Sixes", ascending=False).head()
top_fours = df[["Player", "Fours"]].sort_values(by="Fours", ascending=False).head()
top_5w = df[["Player", "FiveW"]].sort_values(by="FiveW", ascending=False).head()
top_4w = df[["Player", "FourW"]].sort_values(by="FourW", ascending=False).head()

print("\nTop 5 Six Hitters:\n", top_sixes)
print("\nTop 5 Four Hitters:\n", top_fours)
print("\nTop 5 FiveW hauls:\n", top_5w)
print("\nTop 5 FourW hauls:\n", top_4w)

# ----------------------------
# 1️⃣2️⃣ Highest Strike Rate & Best Economy
# ----------------------------
beststr_idx = df["Strike_Rate"].idxmax()
print("\nHighest Strike Rate:", df.iloc[beststr_idx][["Player", "Strike_Rate"]].to_dict())

best_economy = df["Economy"].replace(0, np.nan).idxmin()
print("Best Bowling Economy:", df.iloc[best_economy][["Player", "Economy"]].to_dict())

# ----------------------------
# 1️⃣3️⃣ Correlation Matrix
# ----------------------------
corr = df[["Runs", "Strike_Rate", "Wickets", "Economy"]].corr()
print("\nCorrelation Matrix:")
print(corr)

# ----------------------------
# 1️⃣4️⃣ Visualizations
# ----------------------------

#Top Run Scorers Bar Chart
top_run_scorers = df[["Player", "Runs"]].sort_values(by="Runs", ascending=False).head(5)

plt.figure(figsize=(10,6))
plt.bar(top_run_scorers["Player"], top_run_scorers["Runs"], color='skyblue')
plt.title("Top 5 Run Scorers in IPL 2025")
plt.xlabel("Player")
plt.ylabel("Runs Scored")
#plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

#Top Wicket Takers Bar Chart
top_wicket_takers = df[["Player", "Wickets"]].sort_values(by="Wickets", ascending=False).head(5)

plt.figure(figsize=(10,6))
plt.bar(top_wicket_takers["Player"], top_wicket_takers["Wickets"], color='salmon')
plt.title("Top 5 Wicket Takers in IPL 2025")
plt.xlabel("Player")
plt.ylabel("Wickets Taken")
#plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Sort players by Runs and Wickets for meaningful top 20 analysis
top20_batting = df.sort_values(by="Runs", ascending=False).head(20)
top20_bowling = df.sort_values(by="Wickets", ascending=False).head(20)

#Top 20 Players With Best Batting Strike Rate
plt.figure(figsize=(14,7))
plt.bar(top20_batting["Player"], top20_batting["Strike_Rate"], color='mediumseagreen', edgecolor='black')
plt.title("Top 20 Players by Runs: Strike Rate in IPL 2025")
plt.xlabel("Player")
plt.ylabel("Strike Rate")
plt.xticks(rotation=90, ha="right", fontsize=10)
plt.tight_layout()
plt.show()

#Top 20 Players With Baet Bowling Economy
plt.figure(figsize=(14,7))
plt.bar(top20_bowling["Player"], top20_bowling["Economy"], color='coral', edgecolor='black')
plt.title("Top 20 Players by Wickets: Economy in IPL 2025")
plt.xlabel("Player")
plt.ylabel("Economy")
plt.xticks(rotation=90, ha="right", fontsize=10)
plt.tight_layout()
plt.show()

#Top 20 Player With Most Fours
plt.figure(figsize=(14,7))
plt.bar(top20_batting["Player"], top20_batting["Fours"], color='skyblue', edgecolor='black')
plt.title("Top 20 Players by Runs: Number of Fours in IPL 2025")
plt.xlabel("Player")
plt.ylabel("Fours")
plt.xticks(rotation=90, ha="right", fontsize=10)
plt.tight_layout()
plt.show()

#Top 20 Players With Most Sixes
plt.figure(figsize=(14,7))
plt.bar(top20_batting["Player"], top20_batting["Sixes"], color='orange', edgecolor='black')
plt.title("Top 20 Players by Runs: Number of Sixes in IPL 2025")
plt.xlabel("Player")
plt.ylabel("Sixes")
plt.xticks(rotation=90, ha="right", fontsize=10)
plt.tight_layout()
plt.show()

#Top 20 players with Most Four-Wicket Hauls
plt.figure(figsize=(14,7))
plt.bar(top20_bowling["Player"], top20_bowling["FourW"], color='plum', edgecolor='black')
plt.title("Top 20 Players by Wickets: Four-Wicket Hauls in IPL 2025")
plt.xlabel("Player")
plt.ylabel("Four-Wicket Hauls")
plt.xticks(rotation=90, ha="right", fontsize=10)
plt.tight_layout()
plt.show()












