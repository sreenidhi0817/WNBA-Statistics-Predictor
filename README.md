# WNBA-Statistics-Predictor
Built a machine learning model to predict WNBA player and team performance using historical stats. Implemented data preprocessing, exploratory analysis, and predictive modeling to forecast outcomes like points, rebounds, and win probabilities with visualized insights.

The goal was to apply regression techniques to forecast player stats. After experimenting with different approaches, the final model used was a Random Forest Regressor wrapped in MultiOutputRegressor.

Step 1: Data Cleaning
- Read player info and game stats from Excel (`LAS.xlsx`).
- Standardized player names and home/away labels.
- Removed games where players did not play (DNP).
- Merged opponent average allowed stats for context (e.g., points allowed, rebounds allowed).

Step 2: Building and Testing the Model
- Selected features: Minutes, Age, Height, Position, Opponent, Home/Away, Opponent Allowed Stats.
- Targets: PTS, REB, AST, STL, BLK.
- Split the data into training and test sets (80/20).
- Built a pipeline with preprocessing (numeric + categorical) and a Random Forest Regressor.
- Evaluated model accuracy using held-out test data.
- 
Step 3: Making Predictions
- Implemented functions to:
  - Predict full stat lines for a player against a chosen opponent.
  - Predict over/under outcomes for specific stats.
- Added an interactive menu for user input.
- Example output:
Predicted Stats: {'PTS': 22, 'REB': 11, 'AST': 3, 'STL': 2, 'BLK': 2}


Note: The dataset currently only includes data for Los Angeles Sparks, last updated August 12.
