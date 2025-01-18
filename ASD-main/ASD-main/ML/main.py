import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# Loading the dataset
df=pd.read_csv('ASD_children_traits.csv')
df.shape

df.drop("CASE_NO_PATIENT'S",axis=1,inplace=True)
df.drop("Ethnicity",axis=1, inplace=True)
df.drop("Who_completed_the_test",axis=1,inplace=True)

df.isnull().sum()

data=df.dropna()

data.isnull().sum()

print(data['Speech Delay/Language Disorder'].unique())
print(data['Learning disorder'].unique())
print(data['Genetic_Disorders'].unique())
print(data['Depression'].unique())
print(data['Global developmental delay/intellectual disability'].unique())
print(data['Social/Behavioural Issues'].unique())
print(data['Anxiety_disorder'].unique())
print(data['Sex'].unique())
print(data['Jaundice'].unique())
print(data['Family_mem_with_ASD'].unique())
print(data['ASD_traits'].unique())


# Label Encoding

from sklearn.preprocessing import LabelEncoder


le_speech_delay = LabelEncoder()
data['Speech Delay/Language Disorder'] = le_speech_delay.fit_transform(data['Speech Delay/Language Disorder'])


le_learning_disorder = LabelEncoder()
data['Learning disorder'] = le_learning_disorder.fit_transform(data['Learning disorder'])

# Assuming 'Genetic_Disorders' is a separate categorical column
le_genetic_disorders = LabelEncoder()
data['Genetic_Disorders'] = le_genetic_disorders.fit_transform(data['Genetic_Disorders'])

depression = LabelEncoder()
data['Depression'] = depression.fit_transform(data['Depression'])

# Assuming 'Global developmental delay/intellectual disability' is a separate categorical column
le_global_delay = LabelEncoder()
data['Global developmental delay/intellectual disability'] = le_global_delay.fit_transform(data['Global developmental delay/intellectual disability'])


social_behaviour = LabelEncoder()
data['Social/Behavioural Issues'] = social_behaviour.fit_transform(data['Social/Behavioural Issues'])


# Assuming 'Anxiety_disorder' is a separate categorical column
le_anxiety_disorder = LabelEncoder()
data['Anxiety_disorder'] = le_anxiety_disorder.fit_transform(data['Anxiety_disorder'])

# Assuming 'Sex' is a separate categorical column
le_sex = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])

# Assuming 'Jaundice' is a separate categorical column
le_jaundice = LabelEncoder()
data['Jaundice'] = le_jaundice.fit_transform(data['Jaundice'])

# Assuming 'Family_mem_with_ASD' is a separate categorical column
le_family_mem_with_asd = LabelEncoder()
data['Family_mem_with_ASD'] = le_family_mem_with_asd.fit_transform(data['Family_mem_with_ASD'])

# Assuming 'ASD_traits' is a separate categorical column
le_asd_traits = LabelEncoder()
data['ASD_traits'] = le_asd_traits.fit_transform(data['ASD_traits'])


print(data['Speech Delay/Language Disorder'].unique())
print(data['Learning disorder'].unique())
print(data['Genetic_Disorders'].unique())
print(data['Depression'].unique())
print(data['Global developmental delay/intellectual disability'].unique())
print(data['Social/Behavioural Issues'].unique())
print(data['Anxiety_disorder'].unique())
print(data['Sex'].unique())   #female->0, male->1
print(data['Jaundice'].unique())
print(data['Family_mem_with_ASD'].unique())
print(data['ASD_traits'].unique())


# DATA VISUALIZATION


key_features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10_Autism_Spectrum_Quotient',
                'Social_Responsiveness_Scale', 'Age_Years', 'Qchat_10_Score', 'Childhood Autism Rating Scale',
                'Speech Delay/Language Disorder', 'Learning disorder', 'Genetic_Disorders', 'Depression',
                'Global developmental delay/intellectual disability', 'Anxiety_disorder', 'Sex',
                'Family_mem_with_ASD', 'ASD_traits']

# Histograms for selected numerical features
numerical_features = [col for col in key_features if df[col].dtype in ['int64', 'float64']]
data[numerical_features].hist(figsize=(12, 10))
plt.suptitle('Histograms of Selected Numerical Features')
plt.show()

# Box plots for selected numerical features
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numerical_features])
plt.title('Box Plot of Selected Numerical Features')
plt.xticks(rotation=45)
plt.show()


# Splitting the dataset

X=data.drop(["ASD_traits"],  axis=1)
y=data["ASD_traits"]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.preprocessing import QuantileTransformer, PowerTransformer, Normalizer, MaxAbsScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings


# Feature Scaling

scalers = {
    'QuantileTransformer': QuantileTransformer(),
    'Normalizer': Normalizer(),
    'MaxAbsScaler': MaxAbsScaler()
}


# Classifiers

classifiers = {
  'AdaBoost': AdaBoostClassifier(),
  'RandomForest': RandomForestClassifier(),
  'LogisticRegression': LogisticRegression(),
  'SVM': SVC(),
  'LDA': LinearDiscriminantAnalysis()
}


# Dictionary to store predictions
predictions = {}

for scaler_name, scaler in scalers.items():
    try:
        # Apply feature scaling
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # Iterate over Classifiers
        for classifier_name, clf in classifiers.items():
            # Train the classifier
            clf.fit(X_train_scaled, y_train)
            
            # Make predictions on test data
            y_pred = clf.predict(X_test_scaled)
            
            
            # Store predictions in the dictionary
            predictions[(scaler_name, classifier_name)] = y_pred
            
            
             # Evaluate performance using accuracy_score or other metrics
            accuracy = accuracy_score(y_test, y_pred)

            print(f"Feature Scaling: {scaler_name}, Classifier: {classifier_name}, Accuracy: {accuracy:.2f}")
    
    except AttributeError:
    # Suppress warnings for PowerTransformer (refer to previous discussions)
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
    
    
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#decision tree classifier
dt_classifier = DecisionTreeClassifier()

# Instantiate grid search
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5)

# Fit grid search to training data
grid_search.fit(X_train, y_train)
pickle.dump(grid_search,open("trait.pkl",'wb'))

# Get best hyperparameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


print(best_params)
print(best_model)

# Make predictions on test data using the best model
y_pred = best_model.predict(X_test)

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)


print(data)
print(X.shape)


