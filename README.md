# Iris Flower Classification (Logistic Regression vs Decision Tree)

## What is this project?
This project teaches a computer to guess the **species of an Iris flower** (setosa, versicolor or virginica) from four measurements. It compares two simple machine learning models to see which one guesses better.

## Dataset
- File: `iris.csv`
- 150 flowers, 50 of each species
- Features: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- Target: `species`

## What the notebook does
1. **Load the data** with Pandas and check it (shape, summary, missing values).
2. **Explore it** with a Seaborn pairplot to see how the species differ.
3. **Split the data** 80% for training and 20% for testing (120 / 30 flowers), keeping the species balanced.
4. **Scale the features** with `StandardScaler` so all measurements are on the same scale. The scaler learns only from the training data.
5. **Train two models:** Logistic Regression and Decision Tree.
6. **Evaluate them** using accuracy, precision, recall and F1-score (classification report).
7. **Plot confusion matrix heatmaps** to see which flowers were guessed wrong.
8. **Plot decision boundaries** (using the two petal features) to see how each model separates the species.

## Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 93.33% |
| Decision Tree | 96.67% |

- Setosa was always correct. The few mistakes were between versicolor and virginica.
- The test set is small (30 flowers), so one wrong flower changes accuracy by about 3.3%.

## Tools used
Python, Pandas, Scikit-Learn, Seaborn, Matplotlib (in a Jupyter Notebook, VS Code).

## How to run
1. Put `iris.csv` and the notebook in the same folder. (make sure that they are in same folder.)
2. Install the libraries:( it is optional step and you don't need to do it if the libraries are already installed in your VS Code.)
   ```
   pip install pandas seaborn matplotlib scikit-learn
   ```
3. Open the notebook in VS Code, select a Python kernel, and click **Run All**.

## Files
- `iris.csv`: the dataset
- `*.ipynb`: the notebook with the code and outputs
- `README.md`: this file