import pandas as pd
from pathlib import Path

DATA_PATH = Path(r"C:\Users\sreen\Desktop\LAS.xlsx")
INFO_PATH = "Player_Info"
STAT_PATH = "Statistics"



df_info = pd.read_excel(DATA_PATH, sheet_name=INFO_PATH, engine="openpyxl")
df_stats = pd.read_excel(DATA_PATH, sheet_name=STAT_PATH, engine="openpyxl")

if "HOME/AWAY" in df_stats.columns:
    df_stats["HOME/AWAY"] = (
        df_stats["HOME/AWAY"]
        .astype(str).str.strip().str.upper()
        .replace({"H": "HOME", "HOME": "HOME", "A": "AWAY", "AWAY": "AWAY"})
    )

if "DNP" in df_stats.columns:
    df_stats = df_stats[df_stats["DNP"] == 0]

df_stats = df_stats.dropna(how="all")

for df in (df_stats, df_info):
    if "NAME" in df.columns:
        df["NAME"] = df["NAME"].astype(str).str.strip()

merged_df = pd.merge(df_stats, df_info, on="NAME", how="left")

stats_ha = df_stats.copy()
stats_ha["HOME/AWAY"] = (
    stats_ha["HOME/AWAY"]
    .astype(str).str.strip().str.upper()
    .replace({"H": "HOME", "HOME": "HOME", "A": "AWAY", "AWAY": "AWAY"})
)

opp_allow_ha = (
    stats_ha.groupby(["VS", "HOME/AWAY"])[["PTS", "REB", "AST", "STL", "BLK"]]
    .mean()
    .rename(columns=lambda c: f"OPP_HA_ALLOW_{c}")
    .reset_index()
)

merged_df = merged_df.merge(opp_allow_ha, on=["VS", "HOME/AWAY"], how="left")

merged_df["HOME_FLAG"] = (merged_df["HOME/AWAY"] == "HOME").astype(int)

opp_cols = ["OPP_HA_ALLOW_PTS", "OPP_HA_ALLOW_REB", "OPP_HA_ALLOW_AST",
            "OPP_HA_ALLOW_STL", "OPP_HA_ALLOW_BLK"]
for col in opp_cols:
    if col in merged_df.columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")
        merged_df[col] = merged_df[col].fillna(merged_df[col].mean())

merged_df = merged_df.dropna(axis=1, how="all")

features = [
    "MIN", "AGE", "HEIGHT(IN)", "POSITION", "HOME/AWAY", "VS",
    "AVG_PTS", "AVG_REB", "AVG_AST", "AVG_STL",
# new home/away-aware opponent concessions:
    "OPP_HA_ALLOW_PTS", "OPP_HA_ALLOW_REB", "OPP_HA_ALLOW_AST",
    "OPP_HA_ALLOW_STL", "OPP_HA_ALLOW_BLK",
    # simple numeric flag for home games:
    "HOME_FLAG",
]
targets = ["PTS", "REB", "AST", "STL", "BLK"]




main_df = merged_df.dropna(subset=[c for c in (features + targets) if c in merged_df.columns])

from sklearn.model_selection import train_test_split

X = main_df[features]
Y = main_df[targets]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

intended_numeric = [
    "MIN", "AGE", "HEIGHT(IN)",
    "OPP_HA_ALLOW_PTS", "OPP_HA_ALLOW_REB", "OPP_HA_ALLOW_AST",
    "OPP_HA_ALLOW_STL", "OPP_HA_ALLOW_BLK",
    "HOME_FLAG",
]
intended_categorical = ["POSITION", "HOME/AWAY", "VS"]

# Only keep features that actually exist in X
numeric_features = [c for c in intended_numeric if c in X.columns]
categorical_features = [c for c in intended_categorical if c in X.columns]



preprocessor = ColumnTransformer(
    transformers=[
        ("numbers", "passthrough", numeric_features),
        ("text", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop",
)



from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)))
])

model.fit(X_train, Y_train)


def predict_from_row(row_dict):
    X_new = pd.DataFrame([row_dict], columns=features)
    Y_hat = model.predict(X_new)[0]
    return {stat: int(round(val)) for stat, val in zip(targets ,Y_hat)}


def predict_by_player(name, overrides=None):
    subset = main_df[main_df["NAME"].str.contains(name, case=False, na=False)]
    if subset.empty:
        raise ValueError(f"Player not found: {name}")

    base = subset.iloc[-1][features].to_dict()

    if overrides is None:
        overrides = {}

    if "VS" not in overrides:
        overrides["VS"] = input("Enter opponent team code (VS): ").strip().upper()

    if "HOME/AWAY" not in overrides:
        ha = input("Home or Away? (HOME/AWAY): ").strip().upper()
        overrides["HOME/AWAY"] = "HOME" if ha.startswith("H") else "AWAY"

    if "HOME_FLAG" not in overrides:
        overrides["HOME_FLAG"] = 1 if overrides["HOME/AWAY"] == "HOME" else 0

    base.update(overrides)
    return predict_from_row(base)


def over_under(name, stat, line, overrides=None):
    stat = stat.upper()
    if stat not in [s.upper() for s in targets]:
        raise ValueError(f"Stat must be one of {targets}")
    preds = predict_by_player(name, overrides=overrides)
    val = preds[[t for t in targets if t.upper() == stat][0]]
    return ("Over", val) if val > float(line) else ("Under", val)

#Interactive Menu

while True:
    print("\n---Prediction Menu---")
    print("1. Predict full stats for a player")
    print("2. Over/Under prediction for a stat")
    print("3. Quit")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        name = input("Enter player name: ").strip()
        overrides = {}

        # Ask for opponent and home/away
        overrides["VS"] = input("Enter opponent team code (VS): ").strip().upper()
        ha = input("Home or Away? (HOME/AWAY): ").strip().upper()
        overrides["HOME/AWAY"] = "HOME" if ha.startswith("H") else "AWAY"

        try:
            result = predict_by_player(name, overrides=overrides)
            print("\nPredicted Stats:", result)
        except ValueError as e:
            print("Error:", e)

    elif choice == "2":
        name = input("Enter player name: ").strip()
        stat = input("Enter which stat (PTS, REB, AST, STL, BLK): ").strip()
        line = input("Enter line value (over/under what number?): ").strip()

        overrides = {}
        overrides["VS"] = input("Enter opponent team code (VS): ").strip().upper()
        ha = input("Home or Away? (HOME/AWAY): ").strip().upper()
        overrides["HOME/AWAY"] = "HOME" if ha.startswith("H") else "AWAY"

        try:
            result, pred_val = over_under(name, stat, float(line), overrides=overrides)
            print(f"\nPrediction: {result} (Predicted {stat.upper()} = {pred_val})")
        except ValueError as e:
            print("Error:", e)


    elif choice == "3":
        print("Exiting...")
        break

    else:
        print("Invalid choice. Please select 1, 2, or 3")









