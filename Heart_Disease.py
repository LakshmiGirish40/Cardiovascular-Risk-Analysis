import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
dataset = pd.read_csv(r"D:\Data_Science&AI\Spyder\Heart_Disease\heart_disease_dataset.csv")
df = dataset.copy()

# Define the continuous features
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
# Filter out continuous features for the univariate analysis
df_continuous = df[continuous_features]

# Set up the subplot
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Loop to plot histograms for each continuous feature
for i, col in enumerate(df_continuous.columns):
    x = i // 3
    y = i % 3
    values, bin_edges = np.histogram(df_continuous[col], 
                                     range=(np.floor(df_continuous[col].min()), np.ceil(df_continuous[col].max())))
    
    graph = sns.histplot(data=df_continuous, x=col, bins=bin_edges, kde=True, ax=ax[x, y],
                         edgecolor='none', color='red', alpha=0.6, line_kws={'lw': 3})
    ax[x, y].set_xlabel(col, fontsize=15)
    ax[x, y].set_ylabel('Count', fontsize=12)
    ax[x, y].set_xticks(np.round(bin_edges, 1))
    ax[x, y].set_xticklabels(ax[x, y].get_xticks(), rotation=45)
    ax[x, y].grid(color='lightgrey')
    
    for j, p in enumerate(graph.patches):
        ax[x, y].annotate('{}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 1),
                          ha='center', fontsize=10, fontweight="bold")
    
    textstr = '\n'.join((
        r'$\mu=%.2f$' % df_continuous[col].mean(),
        r'$\sigma=%.2f$' % df_continuous[col].std()
    ))
    ax[x, y].text(0.75, 0.9, textstr, transform=ax[x, y].transAxes, fontsize=12, verticalalignment='top',
                  color='white', bbox=dict(boxstyle='round', facecolor='#ff826e', edgecolor='white', pad=0.5))

ax[1,2].axis('off')
plt.suptitle('Distribution of Continuous Variables', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
#====================================================

# Filter out categorical features for the univariate analysis
categorical_features = df.columns.difference(continuous_features)
df_categorical = df[categorical_features]
# Set up the subplot for a 4x2 layout
fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 18))

# Loop to plot bar charts for each categorical feature in the 4x2 layout
for i, col in enumerate(categorical_features):
    row = i // 2
    col_idx = i % 2
    
    # Calculate frequency percentages
    value_counts = df[col].value_counts(normalize=True).mul(100).sort_values()
    
    # Plot bar chart
    value_counts.plot(kind='barh', ax=ax[row, col_idx], width=0.8, color='green')
    
    # Add frequency percentages to the bars
    for index, value in enumerate(value_counts):
        ax[row, col_idx].text(value, index, str(round(value, 1)) + '%', fontsize=15, weight='bold', va='center')
    
    ax[row, col_idx].set_xlim([0, 95])
    ax[row, col_idx].set_xlabel('Frequency Percentage', fontsize=12)
    ax[row, col_idx].set_title(f'{col}', fontsize=20)

ax[4,1].axis('off')
plt.suptitle('Distribution of Categorical Variables', fontsize=22)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
#==============================================================================
Q1 = df[continuous_features].quantile(0.25)
Q3 = df[continuous_features].quantile(0.75)
IQR = Q3 - Q1
outliers_count_specified = ((df[continuous_features] < (Q1 - 1.5 * IQR)) | (df[continuous_features] > (Q3 + 1.5 * IQR))).sum()

outliers_count_specified
#==========================================================================
# Define the features (X) and the output labels (y)
X = df.drop('target', axis=1)
y = df['target'] 

df1 = df.copy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Fit K Neighbours Classifier to the training Set
#import KNeighbors classifier from sklearn
from sklearn.neighbors import KNeighborsClassifier
#instance the model
knn = KNeighborsClassifier(n_neighbors=3)

#fit the model to the training set
knn.fit(X_train, y_train)
#predict test-set results
y_pred_knn = knn.predict(X_test)
y_pred_knn

#==========================================================================
# probability of getting output as 2 - benign cancer
predict_proba_0 = knn.predict_proba(X_test)[:,0]

# probability of getting output as  4 - malignant cancer
predict_proba_1 = knn.predict_proba(X_test)[:,1]
#===========================================================================
# Check accuracy score
from sklearn.metrics import accuracy_score
y_pred_knn_acc = print('Model accuracy score:{0:0.4f}'.format(accuracy_score(y_test,y_pred_knn)))
#y_test are the true class labels and y_pred_knn are the predicted class labels in the test_set.
#Model accuracy score:0.8361

#Compare the train-set and test-set accuracy
y_pred_train = knn.predict(X_train)
y_pred_train_acc = print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
#Training-set accuracy score:0.8678

#Check for overfitting and underfitting
# print the scores on training and test set
X_train_y_train_acc = print('Training set score: {:.4f}'.format(knn.score(X_train, y_train)))
X_test_y_test_acc = print('Test set score: {:.4f}'.format(knn.score(X_test, y_test)))
#Training set score: 0.0.8678
#Test set score: 0.8361
#The training-set accuracy score is 0.9821 while the test-set accuracy to be 0.9714. These two values are quite comparable. So, there is no question of overfitting.


#Check for overfitting and underfitting
# print the scores on training and test set
X_train_y_train_acc = print('Training set score: {:.4f}'.format(knn.score(X_train, y_train)))
X_test_y_test_acc = print('Test set score: {:.4f}'.format(knn.score(X_test, y_test)))
#==========================================================
#Rebuild kNN Classification model using different values of k 
# instantiate the model with k=5
knn_5 = KNeighborsClassifier(n_neighbors=5)
# fit the model to the training set
knn_5.fit(X_train, y_train)

knn5_train = X_train.copy()
# predict on the test-set
y_pred_5 = knn_5.predict(X_test)
print('Model accuracy score with k=5 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_5)))
y_pred_knn5_acc = accuracy_score(y_test, y_pred_5)
#Model accuracy score with k=5 :0.8197
#===============================================================================
# instantiate the model with k=6
knn_6 = KNeighborsClassifier(n_neighbors=6)
# fit the model to the training set
knn_6.fit(X_train, y_train)
# predict on the test-set
y_pred_6 = knn_6.predict(X_test)
print('Model accuracy score with k=6 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_6)))
y_pred_knn6_acc = accuracy_score(y_test, y_pred_6)
#Model accuracy score with k=6 :  0.8361
#============================================================================
# instantiate the model with k=7
knn_7 = KNeighborsClassifier(n_neighbors=7)
# fit the model to the training set
knn_7.fit(X_train, y_train)
# predict on the test-set
y_pred_7 = knn_7.predict(X_test)
y_pred_train_knn7 =  knn_7.predict(X_train)

print('Model accuracy score with k=7 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_7)))
y_pred_knn7_acc = accuracy_score(y_test, y_pred_7)

#y_pred_train_knn7.to_csv('KNN_7_Model.csv',index=False)
#Model accuracy score with k=7 :0.8361

#===========================================================================
# instantiate the model with k=8
knn_8 = KNeighborsClassifier(n_neighbors=8)
# fit the model to the training set
knn_8.fit(X_train, y_train)
# predict on the test-set
y_pred_8 = knn_8.predict(X_test)
# Predict on the training set


y_pred_knn8_acc = accuracy_score(y_test, y_pred_8)
print('Model accuracy score with k=8 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_8)))
#Model accuracy score with k=8 :  0.8525
#===========================================================================
# instantiate the model with k=9
knn_9 = KNeighborsClassifier(n_neighbors=9)
# fit the model to the training set
knn_9.fit(X_train, y_train)
knn9_model =X_train.copy()
# predict on the test-set
y_pred_9 = knn_9.predict(X_test)
print('Model accuracy score with k=9 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_9)))
y_pred_knn9_acc = accuracy_score(y_test, y_pred_9)
#Model accuracy score with k=9 :  0.8525
#============================================================================
# Confusion matrix 
# Print the Confusion Matrix with k =3 and slice it into four pieces

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_knn)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])
#========================================================================
# Print the Confusion Matrix with k =7 and slice it into four pieces

cm_7 = confusion_matrix(y_test, y_pred_7)

print('Confusion matrix\n\n', cm_7)

print('\nTrue Positives(TP) = ', cm_7[0,0])

print('\nTrue Negatives(TN) = ', cm_7[1,1])

print('\nFalse Positives(FP) = ', cm_7[0,1])

print('\nFalse Negatives(FN) = ', cm_7[1,0])

#======================================================================
#visualization for Knn with k =7
# visualize confusion matrix with seaborn heatmap

plt.figure(figsize=(6,4))

cm_matrix = pd.DataFrame(data=cm_7, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
#==================================================================
#Classification metrices 
from sklearn.metrics import classification_report
class_report = print(classification_report(y_test, y_pred_7))

#Classification accuracy
TP = cm_7[0,0]
TN = cm_7[1,1]
FP = cm_7[0,1]
FN = cm_7[1,0]

# print classification accuracy
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

#Classification error
# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))

#Percision
# print precision score
precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))

#Recall
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))

#True Positive Rate
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

#False Positive Rate
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

#Specificity
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))

#=========================================================================
#Adjusting the classification threshold level
# print the first 10 predicted probabilities of two classes- 2 and 4
y_pred_prob = knn.predict_proba(X_test)[0:10]
y_pred_prob

# store the probabilities in dataframe

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - benign cancer (2)', 'Prob of - malignant cancer (4)'])
y_pred_prob_df

# print the first 10 predicted probabilities for class 4 - Probability of malignant cancer
knn.predict_proba(X_test)[0:10, 1]

# store the predicted probabilities for class 4 - Probability of malignant cancer
y_pred_1 = knn.predict_proba(X_test)[:, 1]
#=============================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,f1_score
from sklearn.model_selection import GridSearchCV
# Base RandomForest model
rf_base = RandomForestClassifier(random_state=0)

# Parameter grid for hyperparameter tuning
param_grid_rf = {
    'n_estimators': [10, 30, 50, 70, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3],
    'bootstrap': [True, False]
}

# Function to tune hyperparameters
def tune_clf_hyperparameters(clf, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

# Using the function to get the best parameters
best_rf_params, best_rf_score = tune_clf_hyperparameters(rf_base, param_grid_rf, X_train, y_train)
print('RF Optimal Hyperparameters: \n', best_rf_params)

# Create and train the model with the best hyperparameters
best_rf = RandomForestClassifier(**best_rf_params, random_state=0)
best_rf.fit(X_train, y_train)

y_pred_rf = best_rf.predict(X_test)

# Evaluate the optimized model on the train data
print(classification_report(y_train, best_rf.predict(X_train)))
# Evaluate the optimized model on the test data
print(classification_report(y_test, best_rf.predict(X_test)))

randoforest_acc = accuracy_score(y_test,y_pred_rf)

randoforest_f1score = f1_score(y_test,y_pred_rf)

#===============================================

print("Random forest hyper parameter tuning best attributes and values")
rf_base1 = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth=4, min_samples_leaf=1, 
                                 min_samples_split=2, n_estimators=100, random_state=0)
rf_base1.fit(X_train, y_train)
y_pred_rf1 = rf_base1.predict(X_test)
print(classification_report(y_train, rf_base1.predict(X_train)))
# Evaluate the optimized model on the test data
print(classification_report(y_test, rf_base1.predict(X_test)))

randoforest_acc1 = accuracy_score(y_test,y_pred_rf1)


#================================================================================
#SVM Model Building
from sklearn.svm import SVC
svm = SVC()
# Parameter grid for hyperparameter tuning
param_grid_svm = {
    'C': [0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 20],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 0.5, 1, 5],
    'degree': [2, 3, 4],
}

# Function to tune hyperparameters
def tune_clf_hyperparameters(clf, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_estimator_

# Using the function to get the best parameters and model
best_svm_hyperparams, best_svm = tune_clf_hyperparameters(svm, param_grid_svm, X_train, y_train)
print('SVM Optimal Hyperparameters: \n', best_svm_hyperparams)

# Evaluate the optimized model on the train data
print(classification_report(y_train, best_svm.predict(X_train)))
# Evaluate the optimized model on the test data
print(classification_report(y_test, best_svm.predict(X_test)))

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
def evaluate_model(model, X_test, y_test, model_name):
    y_pred_svm = model.predict(X_test)
    
    svm_accuracy = accuracy_score(y_test,     y_pred_svm)
    precision = precision_score(y_test,  y_pred_svm, average='weighted')
    recall = recall_score(y_test, y_pred_svm, average='weighted')
    f1 = f1_score(y_test, y_pred_svm, average='weighted')
    print("Svm Accracy, Precision, Recall,F1 Score")
    print(f"Evaluation Metrics for {model_name}:")
    print(f"Accuracy: {svm_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        'svm_accuracy': svm_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
svm_evaluation = evaluate_model(best_svm, X_test, y_test, 'SVM')
svm_evaluation
#==========================================================================
import pandas as pd

print('Based on Svm Actual Vs  Predicted ')
# Assuming you've already generated y_pred
y_pred_svm = best_svm.predict(X_test)

# Create a DataFrame for comparison
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_svm})

# Display the comparison
print(comparison.head())

print("mismatched predictions")
# Alternatively, to see mismatched predictions
comparison['Correct'] = comparison['Actual'] == comparison['Predicted']
print(comparison[comparison['Correct'] == False])
#======================================================================
import pandas as pd

print('Based on Knn Actual Vs  Predicted ')
# Assuming you've already generated y_pred
y_pred_8 = knn_8.predict(X_test)
# Create a DataFrame for comparison
comparison_knn = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_8})

# Display the comparison
print(comparison_knn.head())
print("mismatched predictions")
# Alternatively, to see mismatched predictions
comparison_knn['Correct'] =comparison_knn['Actual'] ==comparison_knn['Predicted']
print(comparison_knn[comparison_knn['Correct'] == False])
#====================================================
print('Based on randomforest Actual Vs  Predicted ')
# Assuming you've already generated y_pred
y_pred_rf1 = rf_base1.predict(X_test)
# Create a DataFrame for comparison
comparison_rf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf1})

# Display the comparison
print(comparison_rf.head())
print("mismatched predictions")
# Alternatively, to see mismatched predictions
comparison_rf['Correct'] =comparison_rf['Actual'] == comparison_rf['Predicted']
print(comparison_rf[comparison_rf['Correct'] == False])

# Generate prediction

# Calculate F1 score
f1 = f1_score(y_test,y_pred_rf1, average='weighted')
print(f'F1 Score: {f1}')
#=========================================================================
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Generate predictions
y_pred_rf1 = rf_base1.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test,y_pred_rf1)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification report
print(classification_report(y_test,y_pred_rf1))

"""True Negative (Actual 0, Predicted 0): 21

False Positive (Actual 0, Predicted 1): 6

False Negative (Actual 1, Predicted 0): 3

True Positive (Actual 1, Predicted 1): 31"""



