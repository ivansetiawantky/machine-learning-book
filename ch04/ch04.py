# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: pymlbook
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Machine Learning with PyTorch and Scikit-Learn  
# # -- Code Examples

# %% [markdown]
# ## Package version checks

# %% [markdown]
# Add folder to path in order to load from the check_packages.py script:

# %%
import sys

sys.path.insert(0, "..")

# %% [markdown]
# Check recommended package versions:

# %%
from python_environment_check import check_packages


d = {"numpy": "1.21.2", "matplotlib": "3.4.3", "sklearn": "1.0", "pandas": "1.3.2"}
check_packages(d)

# %% [markdown]
# # Chapter 4 - Building Good Training Datasets – Data Preprocessing

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# ### Overview

# %% [markdown]
# - [Dealing with missing data](#Dealing-with-missing-data)
#   - [Identifying missing values in tabular data](#Identifying-missing-values-in-tabular-data)
#   - [Eliminating training examples or features with missing values](#Eliminating-training-examples-or-features-with-missing-values)
#   - [Imputing missing values](#Imputing-missing-values)
#   - [Understanding the scikit-learn estimator API](#Understanding-the-scikit-learn-estimator-API)
# - [Handling categorical data](#Handling-categorical-data)
#   - [Nominal and ordinal features](#Nominal-and-ordinal-features)
#   - [Mapping ordinal features](#Mapping-ordinal-features)
#   - [Encoding class labels](#Encoding-class-labels)
#   - [Performing one-hot encoding on nominal features](#Performing-one-hot-encoding-on-nominal-features)
# - [Partitioning a dataset into a separate training and test set](#Partitioning-a-dataset-into-seperate-training-and-test-sets)
# - [Bringing features onto the same scale](#Bringing-features-onto-the-same-scale)
# - [Selecting meaningful features](#Selecting-meaningful-features)
#   - [L1 and L2 regularization as penalties against model complexity](#L1-and-L2-regularization-as-penalties-against-model-omplexity)
#   - [A geometric interpretation of L2 regularization](#A-geometric-interpretation-of-L2-regularization)
#   - [Sparse solutions with L1 regularization](#Sparse-solutions-with-L1-regularization)
#   - [Sequential feature selection algorithms](#Sequential-feature-selection-algorithms)
# - [Assessing feature importance with Random Forests](#Assessing-feature-importance-with-Random-Forests)
# - [Summary](#Summary)

# %% [markdown]
# <br>
# <br>

# %%
from IPython.display import Image
# %matplotlib inline

# %% [markdown]
# # Dealing with missing data

# %% [markdown]
# ## Identifying missing values in tabular data

# %%
import pandas as pd
from io import StringIO
import sys

csv_data = """A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,"""

# If you are using Python 2.7, you need
# to convert the string to unicode:

if sys.version_info < (3, 0):
    csv_data = unicode(csv_data)

df = pd.read_csv(StringIO(csv_data))
df

# %%
df.isnull().sum()

# %%
# access the underlying NumPy array
# via the `values` attribute
print(df.values)
print(type(df))
print(type(df.values))

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# ## Eliminating training examples or features with missing values

# %%
# remove rows that contain missing values

df.dropna(axis=0)

# %%
# remove columns that contain missing values

df.dropna(axis=1)

# %%
# only drop rows where all columns are NaN

df.dropna(how="all")

# %%
# drop rows that have fewer than 3 real values

df.dropna(thresh=4)

# %%
# only drop rows where NaN appear in specific columns (here: 'C')

df.dropna(subset=["C"])

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# ## Imputing missing values

# %%
# again: our original array
df.values

# %%
# impute missing values via the column mean

from sklearn.impute import SimpleImputer
import numpy as np

imr = SimpleImputer(missing_values=np.nan, strategy="mean")
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data

# %% [markdown]
# <br>
# <br>

# %%
df.fillna(df.mean())

# %% [markdown]
# ## Understanding the scikit-learn estimator API

# %%
Image(filename="figures/04_02.png", width=400)

# %%
Image(filename="figures/04_03.png", width=300)

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# # Handling categorical data

# %% [markdown]
# ## Nominal and ordinal features

# %%
import pandas as pd

df = pd.DataFrame(
    [
        ["green", "M", 10.1, "class2"],
        ["red", "L", 13.5, "class1"],
        ["blue", "XL", 15.3, "class2"],
    ]
)

df.columns = ["color", "size", "price", "classlabel"]
df

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# ## Mapping ordinal features

# %%
size_mapping = {"XL": 3, "L": 2, "M": 1}

df["size"] = df["size"].map(size_mapping)
df

# %%
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df["size"].map(inv_size_mapping)

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# ## Encoding class labels

# %%
import numpy as np

# create a mapping dict
# to convert class labels from strings to integers
class_mapping = {label: idx for idx, label in enumerate(np.unique(df["classlabel"]))}
class_mapping

# %%
# to convert class labels from strings to integers
df["classlabel"] = df["classlabel"].map(class_mapping)
df

# %%
# reverse the class label mapping
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df["classlabel"] = df["classlabel"].map(inv_class_mapping)
df

# %%
from sklearn.preprocessing import LabelEncoder

# Label encoding with sklearn's LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df["classlabel"].values)
y

# %%
# reverse mapping
class_le.inverse_transform(y)

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# ## Performing one-hot encoding on nominal features

# %%
X = df[["color", "size", "price"]].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X

# %%
from sklearn.preprocessing import OneHotEncoder

X = df[["color", "size", "price"]].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

# %%
from sklearn.compose import ColumnTransformer

X = df[["color", "size", "price"]].values
c_transf = ColumnTransformer(
    [("onehot", OneHotEncoder(), [0]), ("nothing", "passthrough", [1, 2])]
)
c_transf.fit_transform(X).astype(float)

# %%
# one-hot encoding via pandas

pd.get_dummies(df[["price", "color", "size"]])

# %%
# multicollinearity guard in get_dummies

pd.get_dummies(df[["price", "color", "size"]], drop_first=True)

# %%
# multicollinearity guard for the OneHotEncoder

color_ohe = OneHotEncoder(categories="auto", drop="first")
c_transf = ColumnTransformer(
    [("onehot", color_ohe, [0]), ("nothing", "passthrough", [1, 2])]
)
c_transf.fit_transform(X).astype(float)

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# ## Optional: Encoding Ordinal Features

# %% [markdown]
# If we are unsure about the numerical differences between the categories of ordinal features, or the difference between two ordinal values is not defined, we can also encode them using a threshold encoding with 0/1 values. For example, we can split the feature "size" with values M, L, and XL into two new features "x > M" and "x > L". Let's consider the original DataFrame:

# %%
df = pd.DataFrame(
    [
        ["green", "M", 10.1, "class2"],
        ["red", "L", 13.5, "class1"],
        ["blue", "XL", 15.3, "class2"],
    ]
)

df.columns = ["color", "size", "price", "classlabel"]
df

# %% [markdown]
# We can use the `apply` method of pandas' DataFrames to write custom lambda expressions in order to encode these variables using the value-threshold approach:

# %%
df["x > M"] = df["size"].apply(lambda x: 1 if x in {"L", "XL"} else 0)
df["x > L"] = df["size"].apply(lambda x: 1 if x == "XL" else 0)

del df["size"]
df

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# # Partitioning a dataset into a separate training and test set

# %%
df_wine = pd.read_csv(
    "https://archive.ics.uci.edu/" "ml/machine-learning-databases/wine/wine.data",
    header=None,
)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df_wine = pd.read_csv('wine.data', header=None)


df_wine.columns = [
    "Class label",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]

print("Class labels", np.unique(df_wine["Class label"]))
print(df_wine.head())
print(df_wine["Class label"].size)
print(df_wine.size)
print(14 * 178)

# %%
from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

print(X_train.size)
print(X_test.size)
print(y_train.size)
print(y_test.size)

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# # Bringing features onto the same scale

# %%
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# %%
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# %% [markdown]
# A visual example:

# %%
ex = np.array([0, 1, 2, 3, 4, 5])

print("standardized:", (ex - ex.mean()) / ex.std())

# Please note that pandas uses ddof=1 (sample standard deviation)
# by default, whereas NumPy's std method and the StandardScaler
# uses ddof=0 (population standard deviation)

# normalize
print("normalized:", (ex - ex.min()) / (ex.max() - ex.min()))

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# # Selecting meaningful features

# %% [markdown]
# ...

# %% [markdown]
# ## L1 and L2 regularization as penalties against model complexity

# %% [markdown]
# ## A geometric interpretation of L2 regularization

# %%
Image(filename="figures/04_05.png", width=500)

# %%
Image(filename="figures/04_06.png", width=500)

# %% [markdown]
# ## Sparse solutions with L1-regularization

# %%
Image(filename="figures/04_07.png", width=500)

# %% [markdown]
# For regularized models in scikit-learn that support L1 regularization, we can simply set the `penalty` parameter to `'l1'` to obtain a sparse solution:

# %%
from sklearn.linear_model import LogisticRegression

LogisticRegression(penalty="l1")

# %% [markdown]
# Applied to the standardized Wine data ...

# %%
from sklearn.linear_model import LogisticRegression

# from sklearn.multiclass import OneVsRestClassifier

lr = LogisticRegression(penalty="l1", C=1.0, solver="liblinear", multi_class="ovr")
# lr = OneVsRestClassifier(LogisticRegression(penalty="l1", C=1.0, solver="liblinear"))
# Note that C=1.0 is the default. You can increase
# or decrease it to make the regulariztion effect
# weaker or stronger, respectively.
lr.fit(X_train_std, y_train)
print("Training accuracy:", lr.score(X_train_std, y_train))
print("Test accuracy:", lr.score(X_test_std, y_test))

# %%
lr.intercept_

# %%
np.set_printoptions(8)

# %%
lr.coef_[lr.coef_ != 0].shape

# %%
lr.coef_

# %%
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)

colors = [
    "blue",
    "green",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "pink",
    "lightgreen",
    "lightblue",
    "gray",
    "indigo",
    "orange",
]

weights, params = [], []
for c in np.arange(-4.0, 6.0):
    lr = LogisticRegression(
        penalty="l1", C=10.0**c, solver="liblinear", multi_class="ovr", random_state=0
    )
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column + 1], color=color)
plt.axhline(0, color="black", linestyle="--", linewidth=3)
plt.xlim([10 ** (-5), 10**5])
plt.ylabel("Weight coefficient")
plt.xlabel("C (inverse regularization strength)")
plt.xscale("log")
plt.legend(loc="upper left")
ax.legend(loc="upper center", bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)

# plt.savefig('figures/04_08.png', dpi=300,
#            bbox_inches='tight', pad_inches=0.2)

plt.show()

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# ## Sequential feature selection algorithms

# %%
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SBS:
    def __init__(
        self,
        estimator,
        k_features,
        scoring=accuracy_score,
        test_size=0.25,
        random_state=1,
    ):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


# %%
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker="o")
plt.ylim([0.7, 1.02])
plt.ylabel("Accuracy")
plt.xlabel("Number of features")
plt.grid()
plt.tight_layout()
# plt.savefig('figures/04_09.png', dpi=300)
plt.show()

# %%
k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

# %%
knn.fit(X_train_std, y_train)
print("Training accuracy:", knn.score(X_train_std, y_train))
print("Test accuracy:", knn.score(X_test_std, y_test))

# %%
knn.fit(X_train_std[:, k3], y_train)
print("Training accuracy:", knn.score(X_train_std[:, k3], y_train))
print("Test accuracy:", knn.score(X_test_std[:, k3], y_test))

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# # Assessing feature importance with Random Forests

# %%
from sklearn.ensemble import RandomForestClassifier

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500, random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print(
        "%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]])
    )

plt.title("Feature importance")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")

plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
# plt.savefig('figures/04_10.png', dpi=300)
plt.show()

# %%
from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print("Number of features that meet this threshold criterion:", X_selected.shape[1])

# %% [markdown]
# Now, let's print the 3 features that met the threshold criterion for feature selection that we set earlier (note that this code snippet does not appear in the actual book but was added to this notebook later for illustrative purposes):

# %%
for f in range(X_selected.shape[1]):
    print(
        "%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]])
    )

# %% [markdown]
# <br>
# <br>

# %% [markdown]
# # Summary

# %% [markdown]
# ...

# %% [markdown]
# ---
#
# Readers may ignore the next cell.

# %%
# # ! python ../.convert_notebook_to_script.py --input ch04.ipynb --output ch04.py
#
# Use jupytext to convert ipynb to py, as below. OR just use vscode Export
# ! jupytext --to py:percent ch04.ipynb
