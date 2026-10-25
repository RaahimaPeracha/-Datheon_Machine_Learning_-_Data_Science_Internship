# California Housing Price Prediction (OLS vs Ridge vs Lasso)

## What is this project?
This project predicts the **median house value** of a neighborhood in California using information like income, location, house age and number of rooms. It compares three linear regression models to see which one predicts best.

## Dataset
- File: `housing.csv`
- 20,640 rows (one row = one block of houses)
- Features: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`
- Target: `median_house_value`
- Only `total_bedrooms` has missing values (207 rows)

## What the notebook does
1. **Load and inspect** the data with Pandas.
2. **Explore it:** target distribution and a correlation heatmap.
3. **Create new features:** ratios (rooms per household, bedrooms per room, population per household) and interactions (income x age, income x rooms).
4. **Fix multicollinearity:** check VIF, then replace the raw count columns (rooms, bedrooms, population, households) with the ratio features.
5. **Encode** the text column `ocean_proximity` with one-hot encoding.
6. **Split the data** 80% train and 20% test.
7. **Build pipelines:** fill missing values with the median, scale features with `StandardScaler`, then fit the model.
8. **Train three models:** OLS, Ridge (L2) and Lasso (L1). Ridge and Lasso are tuned with `GridSearchCV` over many alpha values.
9. **Compare them** using RMSE, MAE and R².
10. **Plot** predicted vs actual values, residual distributions, residuals vs predicted, and model coefficients.

## Results
| Model | RMSE | MAE | R² |
|---|---|---|---|
| OLS | 71,558 | 51,255 | 0.609 |
| Ridge (alpha = 10) | 71,562 | 51,233 | 0.609 |
| Lasso (alpha = 31.6) | 71,565 | 51,236 | 0.609 |

- All three models perform almost the same. Regularization helps little here because there is plenty of data and the multicollinearity was already fixed.
- Lasso did not remove any feature.
- House prices are capped at about 500,000, which creates a diagonal line in the residual plot and limits how well a linear model can do.
- To improve further, try a non-linear model (such as Random Forest) or a log transform of the target.

## Tools used
Python, Pandas, NumPy, Scikit-Learn (LinearRegression, Ridge, Lasso, GridSearchCV, Pipeline), Seaborn, Matplotlib (Jupyter Notebook in VS Code).

## How to run
1. Put `housing.csv` and the notebook in the same folder.
2. Install the libraries:
   ```
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```
3. Open the notebook in VS Code, select a Python kernel, and click **Run All**.

## Files
- `housing.csv`: the dataset
- `housing.ipynb`: the notebook with code, outputs and plots
- `README.md`: this file