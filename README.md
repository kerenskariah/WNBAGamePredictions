# WiCSxSA: Predicting Game Outcomes

This project analyzes regular season game data and advanced stats to predict game outcomes. The workflow leverages data preprocessing, feature selection, and machine learning techniques to build a predictive model.

## Project Overview

1. **Data Loading**: Regular season schedules and advanced stats data are loaded and cleaned.
2. **Data Preprocessing**: Merging datasets, handling missing values, and scaling features.
3. **Feature Selection**: Using a sequential feature selector to pick the most relevant features.
4. **Model Training and Evaluation**: A Monte Carlo simulation evaluates a logistic regression model's accuracy in predicting game outcomes.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook (or Google Colab)
- Libraries: `pandas`, `scikit-learn`, `numpy`

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/WiCSxSA.git
cd WiCSxSA
pip install -r requirements.txt
```

### Dataset

Ensure `reg_season.csv` and `advanced_stats.csv` are in the root directory of your project or modify the file paths in the code as needed.

## Usage

To execute the project:

1. Open file in Jupyter Notebook or Google Colab.
2. Run each cell to preprocess the data, select features, and train the model.
3. The final cell outputs the accuracy of the model after performing a Monte Carlo simulation.

## Code Walkthrough

### Data Loading

```python
import pandas as pd

schedule = pd.read_csv("reg_season.csv")
advanced_stats = pd.read_csv("advanced_stats.csv")
```

### Data Preprocessing

- Drop unnecessary columns and NaN values.
- Merge datasets to consolidate information.
- Normalize data using `MinMaxScaler`.

### Feature Selection

Using `SequentialFeatureSelector` with `RidgeClassifier` to pick the top 10 predictors for game outcomes.

```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
```

### Model Training and Evaluation

A logistic regression model is evaluated with a Monte Carlo simulation over 1000 iterations to calculate the average accuracy.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### Example Output

```
Accuracy: 0.68
```

## Project Structure

- `wnbagamepredictions.py`: Main notebook containing all steps from data loading to model evaluation.

## Contributing

Feel free to open issues or pull requests. Contributions are welcome!

## Acknowledgments

This project was created with support from the WiCS (Women in Computer Science) and ASA (Aggie Sports Analytics) initiatives.
```
