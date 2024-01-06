# Import necessary libraries for data analysis and machine learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV



# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('file:///Users/ahmed777/Desktop/VM/crypto.csv')


print("start of the preprocess")
# Drop unnecessary columns
df.drop(['unix', 'symbol', 'Volume BTC'], axis=1, inplace=True)

print("#####describe the data set:#######\n")
df.describe()

# Checking and removing rows with more than 2 missing values
missing_values_per_row = df.isnull().sum(axis=1)
df = df[missing_values_per_row <= 2]

# Applying forward fill to handle 1 or 2 missing values in remaining rows
df.ffill(inplace=True)


# Display the first few rows of the dataframe
print(df.head())
# Display the shape of the dataframe
print(df.shape)
# Display descriptive statistics
print(df.describe())
# Display information about the dataframe
print(df.info())

print("describe the data set:")
df.describe()


# Check for missing values
print("Check if data is null:")
print(df.isnull().sum())


print("######End Of Preprocessing########\n")


# Define the features to analyze
features = ['Open', 'High', 'Low', 'Close']

print("Plots so we can understand bitcoin price movement\n")



plt.figure(figsize=(15, 5))
plt.plot(df['Close'])
plt.title('Bitcoin Close Price', fontsize=15)
plt.ylabel('Price in Dollars')
plt.show()



# Convert 'Date' to datetime and extract year, month, and day
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day


#start of analyzing bitcoin price movement so we understand the featers 
print("\n start understanding price movement and prediction \n")



df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 0, 1)
print(df.head())


# How many time the price went up or down
print("\n How many time the price went up or down \n")
print(df.groupby('target').size())

# Display the proportion of target variable values
print("\nHow many time the price went up or down in % plot\n")
plt.pie(df['target'].value_counts().values,
        labels=["Goes Down", "Goes Up"], autopct='%1.1f%%')
plt.show()



# Visualize correlations using a heatmap
plt.figure(figsize=(10, 6))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()




# Prepare features and target for machine learning
features = df[['open-close', 'low-high', 'is_quarter_end']]
#features = df[['Open', 'High', 'Low', 'Close']].astype(float) I got low predictions and over fitting with these
target = df['target']

#new test


#end test

# Scatter plot of 'open' vs 'low'
plt.figure(figsize=(8, 5))
plt.scatter(df['target'], df['Close'], alpha=0.6)
plt.title('target vs Close')
plt.xlabel('target')
plt.ylabel('Close')
plt.grid(True)
plt.show()


# Scatter plot of 'High' vs 'low'
plt.figure(figsize=(8, 5))
plt.scatter(df['target'], df['Low'], alpha=0.6)
plt.title('target vs Low')
plt.xlabel('target')
plt.ylabel('Low')
plt.grid(True)
plt.show()





print("# Plot 'open-close' feature against the target \n")
# Plot 'open-close' feature against the target
plt.figure(figsize=(8, 5))
plt.scatter(df['open-close'], target, alpha=0.6)
plt.title('open-close vs Target')
plt.ylabel('open-close')
plt.xlabel('Target')
plt.tight_layout()
plt.show()



print("# Plot 'low-high' feature against the target \n")
# Plot 'open-close' feature against the target
plt.figure(figsize=(8, 5))
plt.scatter(df['low-high'], target, alpha=0.6)
plt.title('low-high vs Target')
plt.xlabel('low-high')
plt.ylabel('Target')
plt.tight_layout()
plt.show()

# Display descriptive statistics
print("Descriptive Statistics of the Dataset:")
print(df.describe())






# Scale features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and validation sets
X_train, X_test, Y_train, Y_test = train_test_split(
    features, target, test_size=0.1, random_state=42)


# Prepare to plot ROC curves
plt.figure(figsize=(10, 8))






### Find the best depth for Decision Tree
print("###########   Plot to help us to find the best Depth of a Tree\n")

# Range of depths to try
depth_range = range(1, 20)

train_scores, test_scores = validation_curve(
    DecisionTreeClassifier(), X_train, Y_train, param_name="max_depth", 
    param_range=depth_range, scoring="accuracy", cv=5
)

# Calculate mean and standard deviation for training set scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

# Calculate mean and standard deviation for test set scores
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot mean accuracy scores for training and test sets
plt.plot(depth_range, train_mean, label="Training score", color="r")
plt.plot(depth_range, test_mean, label="Cross-validation score", color="g")

# Plot accurancy bands for training and test sets
plt.fill_between(depth_range, train_mean - train_std, train_mean + train_std, color="r", alpha=0.2)
plt.fill_between(depth_range, test_mean - test_std, test_mean + test_std, color="g", alpha=0.2)

# Create plot
plt.title("Validation Curve With Decision Tree")
plt.xlabel("Depth")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="best")
plt.show()
print("### End of the best depth for Decision Tree\n")



# Function to find the best k value for KNN
def find_best_k(X_train, Y_train, k_range):
    k_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='roc_auc')
        k_scores.append(scores.mean())

    best_k = k_range[k_scores.index(max(k_scores))]
    return best_k


# Range of k values to test
k_range = range(1, 31)

best_k = find_best_k(X_train, Y_train, k_range)
print(f"The best k value is: {best_k}")



print("###### FIND THE BEST PARAMETERS FOR RANDUM FOREST CLASSIFIER#######")

param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
    'bootstrap': [True, False]  # Method of selecting samples for training each tree
}

# Initialize the classifier
#rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
#grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the GridSearchCV
#grid_search.fit(X_train, Y_train)

# Get the best estimator
#best_rf = grid_search.best_estimator_

# Print the best parameters
#print("Best parameters found: ", grid_search.best_params_)





print("###  HERE WHERE I START TESTING THE MODELS\n")
#models = [LogisticRegression(), SVC(probability=True), XGBClassifier(), RandomForestClassifier(), KNeighborsClassifier(n_neighbors=best_k)]
#model_names = ["Logistic Regression", "SVC", "XGBClassifier", "RandomForestClassifier", "KNN"]

models = [
    LogisticRegression(), 
    XGBClassifier(), 
        RandomForestClassifier(
        n_estimators=200,
        max_depth=None, 
        min_samples_split=10,
        min_samples_leaf=1,
        max_features='auto',
        bootstrap=True
    ), 
    KNeighborsClassifier(n_neighbors=best_k),  # Replace best_k with your value
    DecisionTreeClassifier(max_depth=3)  # Added Decision Tree Classifier
]
model_names = ["Logistic Regression", "XGBClassifier", "RandomForestClassifier", "KNN", "Decision Tree"]


# Display the direction of Bitcoin price movement
print('\n0 : Goes Up')
print('1 : Goes Down')


# Initialize KFold
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Function to evaluate models using K-fold cross-validation
def evaluate_model_accuracy(models, model_names, features, target, kf):
   for i, model in enumerate(models):
        train_acc_scores = []
        test_acc_scores = []
        classification_reports = []
        

        for train_index, test_index in kf.split(features):
            # Split data into training and test sets
            X_train, X_test = features[train_index], features[test_index]
            Y_train, Y_test = target[train_index], target[test_index]

            # Train the model
            model.fit(X_train, Y_train)

            # Make predictions
            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)

            # Calculate accuracy
            train_acc = accuracy_score(Y_train, Y_train_pred)
            test_acc = accuracy_score(Y_test, Y_test_pred)

            # Append accuracy scores
            train_acc_scores.append(train_acc)
            test_acc_scores.append(test_acc)
                       

                # Plotting accuracies for each fold
        folds = np.arange(1, kf.get_n_splits() + 1)
        plt.plot(folds, train_acc_scores, marker='o', label=f'{model_names[i]} - Training')
        plt.plot(folds, test_acc_scores, marker='x', linestyle='--', label=f'{model_names[i]} - Test')

        # Print the average training and test accuracy across all folds
        print(f"{model_names[i]}:")
        print(f"Average Training Accuracy: {np.mean(train_acc_scores):.4f}")
        print(f"Average Test Accuracy: {np.mean(test_acc_scores):.4f}")
        print(f"Standard Deviation of Test Accuracy: {np.std(test_acc_scores):.4f}")
        print("\n")

       
        # Print classification report for the last fold
        print(f"\nClassification Report for {model_names[i]}:")
        print(classification_report(Y_test, Y_test_pred))
        print("\n")
        

          # Set plot titles and labels outside the inner loop
   plt.title('Model Accuracies across K-Fold Cross-Validation')
   plt.xlabel('Fold Number')
   plt.ylabel('Accuracy')
   plt.legend()
   plt.show()

# Call the function to evaluate models
evaluate_model_accuracy(models, model_names, features, target, kf)






# Iterate through the models and plot their confusion matrices
for i, model in enumerate(models):
    # Predicting the test set results
    Y_pred = model.predict(X_test)

    # Plotting the confusion matrix
    ConfusionMatrixDisplay.from_predictions(Y_test, Y_pred)
    plt.title(f'Confusion Matrix for {model_names[i]}')
    plt.show()
    print("\n")



    print("####### ROC_AUC & CURVE")
    # Train each model, print accuracy, and check for overfitting
for i, model in enumerate(models):
    model.fit(X_train, Y_train)
    
    # Prediction and accuracy
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Calculate and print ROC AUC scores
    train_acc = metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:,1])
    valid_acc = metrics.roc_auc_score(Y_test, model.predict_proba(X_test)[:,1])
    print(f"{model_names[i]} : ")

    print()
    print("Training ROC AUC : ", train_acc)
    print("Validation ROC AUC : ", valid_acc)




# Plot ROC curves
for model, name in zip(models, model_names):
    proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(Y_test, proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot ROC curve for a random classifier for comparison
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance (AUC = 0.50)', alpha=.8)

# Adding customizations
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Show plot
plt.show()
print("#########  The best result that i got was with RANDUM FOREST with Average Test Accuracy: 0.9838 #############")
print("######### I GOT SO BAD RESULTS WITH Logistic Regression AND SVC#############")
#i had an issue with overfitting and i changed the randum state from 2022 to 41 and it worked