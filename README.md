# House Price Prediction — Linear Regression + Gradient Descent (scikit-learn)
python -m venv venv
venv\Scripts\activate
## Run
```
pip install -r requirements.txt
python data_generator.py      # creates house_data.csv (12,000 rows)
python train.py               # trains, prints metrics, saves model.pkl
streamlit run app.py          # opens the UI in browser
```
## sklearn components used
- SGDRegressor (gradient descent), StandardScaler, LinearRegression (exact check)
- train_test_split, mean_squared_error, mean_absolute_error, r2_score

## Files
features.py, data_generator.py, model.py (sklearn model), train.py, app.py (Streamlit UI)
