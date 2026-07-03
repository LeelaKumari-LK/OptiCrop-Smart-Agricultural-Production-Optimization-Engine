import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 8)

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print("All libraries imported successfully!")

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Basic exploration
print("First 5 rows:")
print(df.head())

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nColumn info & data types:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

print("\nUnique crops in the dataset:")
print(df['label'].unique())

print("\nNumber of unique crops:")
print(df['label'].nunique())

print("\nCount of samples per crop:")
print(df['label'].value_counts())


# ===== UNIVARIATE ANALYSIS =====

numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Distribution plots for each numeric feature
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=30, color='green')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Countplot for the target variable (crop labels)
plt.figure(figsize=(15, 8))
sns.countplot(x='label', data=df)
plt.title('Count of Samples per Crop')
plt.xlabel('Crop')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# ===== BIVARIATE ANALYSIS: Humidity vs Crop Label =====

plt.figure(figsize=(15, 8))
sns.stripplot(x='label', y='humidity', data=df, jitter=True)
plt.title('Humidity Distribution Across Different Crops')
plt.xlabel('Crop')
plt.ylabel('Humidity (%)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.scatterplot(x='temperature', y='humidity', size='rainfall', hue='label', data=df, palette='tab20', legend=False)
plt.title('Temperature vs Humidity vs Rainfall vs Crop (Multivariate View)')
plt.tight_layout()
plt.show()


# ===== MULTIVARIATE ANALYSIS: Combined Bar Plot =====

df_melted = df.melt(value_vars=numeric_cols, var_name='Feature', value_name='Value')

plt.figure(figsize=(12, 8))
sns.barplot(x='Feature', y='Value', data=df_melted, estimator=np.mean, errorbar=None, hue='Feature', palette='viridis', legend=False)
plt.title('Average Value of Agricultural Features (N, P, K, Temperature, Humidity, pH, Rainfall)')
plt.xlabel('Feature')
plt.ylabel('Average Value')
plt.tight_layout()
plt.show()


# ===== EPIC 3: DATA PRE-PROCESSING =====
# Story 1: Check for Null Values

print("\nNull values per column:")
print(df.isnull().sum())

print("\nTotal null values in dataset:", df.isnull().sum().sum())


# Story 2: Detect and Treat Outliers

print("\n--- Outlier Detection (IQR method) ---")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers (bounds: {lower_bound:.2f} to {upper_bound:.2f})")

# Visualize outliers with box plots
plt.figure(figsize=(15, 8))
df[numeric_cols].boxplot()
plt.title('Outlier Detection Across All Numeric Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ===== Story 2: Handling Outliers (Potassium) =====

# Step 1: Visualize K (Potassium) outliers with a boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['K'])
plt.title('Boxplot of Potassium (K) - Outlier Check')
plt.tight_layout()
plt.show()

# Step 2: Calculate IQR bounds for K
Q1 = df['K'].quantile(0.25)
Q3 = df['K'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Potassium (K) - Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")

outliers_k = df[(df['K'] < lower_bound) | (df['K'] > upper_bound)]
print(f"Number of outliers in K: {len(outliers_k)}")

# Step 3: Apply log transformation to normalize K's distribution
df['K'] = np.log1p(df['K'])

# Step 4: Verify the new distribution after transformation
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['K'])
plt.title('Boxplot of Potassium (K) - After Log Transformation')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['K'], kde=True, color='green')
plt.title('Distribution of Potassium (K) - After Log Transformation')
plt.tight_layout()
plt.show()

# Combined boxplot - all features at once
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_cols])
plt.title('Boxplot of All Features - Outlier Overview')
plt.tight_layout()
plt.show()


# ===== Removing Phosphorous Outliers =====

Q1 = df['P'].quantile(0.25)
Q3 = df['P'].quantile(0.75)
IQR = Q3 - Q1

filter_p = (df['P'] >= Q1 - 1.5 * IQR) & (df['P'] <= Q3 + 1.5 * IQR)
df = df.loc[filter_p]

print("Shape after removing phosphorous outliers:", df.shape)


# ===== Story 3: Extracting Seasonal Crops =====

print("Summer Crops")
print(df[(df['temperature'] > 30) & (df['humidity'] > 50)]['label'].unique())
print("----------------------------")

print("Winter Crops")
print(df[(df['temperature'] < 20) & (df['humidity'] > 30)]['label'].unique())
print("----------------------------")

print("Rainy Crops")
print(df[(df['rainfall'] > 200) & (df['humidity'] > 50)]['label'].unique())
print("----------------------------")


# ===== Story 4: Splitting Data into Train and Test Sets =====

y = df['label']
x = df.drop(['label'], axis=1)

print("Shape of x", x.shape)
print("Shape of y", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print("The shape of x train", x_train.shape)
print("The shape of x test", x_test.shape)
print("The shape of y train", y_train.shape)
print("The shape of y test", y_test.shape)


# ===== EPIC 4: MODEL BUILDING =====
# Story 1: K-Means Clustering - Elbow Method

plt.rcParams['figure.figsize'] = (10, 4)
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method", fontsize=20)
plt.xlabel("No of clusters")
plt.ylabel("WCSS")
plt.show()

# Train the K-Means model with optimal clusters (4, based on elbow method)
km = KMeans(n_clusters=4, init="k-means++", max_iter=300, n_init=10, random_state=0)
y_means = km.fit_predict(x)

a = df['label'].reset_index(drop=True)
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis=1)
z = z.rename(columns={0: 'cluster'})

print("Let's check the results after applying the K-Means clustering analysis\n")
print("Crops in First cluster:", z[z['cluster'] == 0]['label'].unique())
print("________________________________________________________")
print("Crops in Second cluster:", z[z['cluster'] == 1]['label'].unique())
print("________________________________________________________")
print("Crops in Third cluster:", z[z['cluster'] == 2]['label'].unique())
print("________________________________________________________")
print("Crops in Fourth cluster:", z[z['cluster'] == 3]['label'].unique())


# ===== Story 2: Logistic Regression =====

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

print("Logistic Regression model trained successfully!")
print("\nFirst 10 predictions:", y_pred[:10])
print("First 10 actual labels:", y_test[:10].values)


# ===== Model Evaluation =====

from sklearn.metrics import classification_report, accuracy_score

cr = classification_report(y_test, y_pred)
print(cr)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.title('Confusion Matrix - Crop Prediction')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# ===== Story 4: Save the Best Model =====

import pickle

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully as 'model.pkl' and 'scaler.pkl'")


# ===== Predict the Best Crop Based on Given Parameters =====

sample_input = pd.DataFrame({
    'N': [105], 'P': [35], 'K': [40],
    'temperature': [25], 'humidity': [64], 'ph': [7], 'rainfall': [160]
})

sample_input['K'] = np.log1p(sample_input['K'])
sample_scaled = scaler.transform(sample_input)

prediction = model.predict(sample_scaled)
print("The suggested crop for given climatic condition is:", prediction[0])