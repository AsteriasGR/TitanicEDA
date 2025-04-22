import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sns.set(style='whitegrid')
plt.rcParams["figure.figsize"] = (10,6)

df = sns.load_dataset("titanic")

#Initial look
print(df.head())
print(df.isnull().sum())
print(df['age'].median())

#Median for age NA
df['age'] = df['age'].fillna(df['age'].median())

print(df['embark_town'].mode()[0])

#First mode for embark_town NA
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

#Too many missing in deck, thus we drop entirely
df = df.drop(columns=['deck'])
#We drop the rest as well, may God have mercy
df.dropna(inplace=True)

#Results of cleaning
print(df.head())
print(df.isnull().sum())

#Feature Engineering
df['family_size'] = df['sibsp'] + df['parch']
df['is_alone'] = (df['family_size'] == 0).astype(int)

#Encoding Categorical to Numerical
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['class'] = df['class'].map({'Third': 3, 'Second': 2, 'First': 1})

#Results of Engineering and Encoding
print(df.head())


sns.barplot(x='sex', y='survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()

sns.barplot(x='pclass', y='survived', data=df)
plt.title('Survival Rate by Class')
plt.show()

sns.histplot(data=df, x='age', hue='survived', kde=True, element='step')
plt.title('Age Distribution by Survival Status')
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Violin plot of age vs survival
sns.violinplot(x='survived', y='age', hue='sex', data=df, split=True)
plt.title('Age Distribution by Survival and Gender')
plt.show()

# Pairplot of numerical features
sns.pairplot(df[['survived', 'age', 'fare', 'family_size']], hue='survived')
plt.suptitle('Pairwise Feature Relationships', y=1.02)
plt.show()

# Select features for modeling
features = ['sex', 'age', 'fare', 'pclass', 'family_size', 'is_alone']
X = df[features]
y = df['survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Results
print(classification_report(y_test, y_pred))

#Model Reading
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(coefficients)
print("Intercept:", model.intercept_)

#Using the Model
predictions = model.predict(X_test)
print(predictions)

probabilities = model.predict_proba(X_test)
print(probabilities[:5])  # Show first 5 rows

#Hypothetical New Passenger
new_passenger = pd.DataFrame([{
    'sex': 1,            # female
    'age': 28,
    'fare': 100,
    'pclass': 1,
    'family_size': 0,
    'is_alone': 1
}])

prediction = model.predict(new_passenger)
probability = model.predict_proba(new_passenger)

print("Prediction (0=did not survive, 1=survived):", prediction[0])
print("Survival probability:", probability[0][1])

# Key Insights:

# Family size matters: Passengers traveling alone had lower survival odds.
# Fare and class are correlated with survival: More expensive tickets, higher class = higher survival.
# Sex remains a strong predictor of survival, as seen in the logistic regression.
# Our basic model already gives good separation using simple engineered features.

# Model Summary:
# This logistic regression model, though basic, shows how even simple features can predict survival. Great starting point for more complex models