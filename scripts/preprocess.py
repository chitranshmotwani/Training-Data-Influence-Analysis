import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load data
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
train_data = pd.read_csv("data/adult.data", header=None, names=column_names, na_values=" ?", skipinitialspace=True)
test_data = pd.read_csv("data/adult.test", header=None, names=column_names, na_values=" ?", skipinitialspace=True, skiprows=1)

# Combine train and test data for preprocessing
data = pd.concat([train_data, test_data], ignore_index=True)

# Handle missing values
data = data.dropna()

# Encode target variable
data['income'] = data['income'].apply(lambda x: 0 if x.strip() == '<=50K' else 1)

# One-hot encode categorical features
categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Normalize numerical features
numerical_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Split into train and test sets
X = data.drop("income", axis=1)
y = data["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)