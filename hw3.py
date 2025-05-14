from pandas import *
from numpy import * 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  train_test_split, GridSearchCV
from sklearn.preprocessing import  LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

#a
#a1
data_frame = read_csv('milling_machine.csv')  
data_frame.info()
print(data_frame.head())
print(data_frame.describe())

#a2
missing_v = data_frame.isnull().sum()
missing_v_ratio = data_frame.isnull().mean() 

missing_data = DataFrame({
    'Missing value Count': missing_v,
    'Missing value Ratio': missing_v_ratio
})
print(missing_data)

#a3
new_data = data_frame.copy()
label_encoder = LabelEncoder()
for column in new_data.select_dtypes(include='object').columns:
    new_data[column] = label_encoder.fit_transform(new_data[column])

correlation_matrix = new_data.corr(numeric_only=True)
correlation_with_failure = correlation_matrix['Failure Types']
sorted_corr = correlation_with_failure.sort_values(ascending=False)
print(sorted_corr)

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#a4
top_features = sorted_corr.drop('Failure Types').abs().nlargest(3).index

for feature in top_features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=data_frame, x=feature, bins=10, kde=False)  
    plt.title(f'Distribution of {feature} (Binned)')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

#b
#b1
for column in data_frame.columns:
    missing_value = data_frame[column].isnull().sum()
    if missing_value > 0:
        if data_frame[column].dtype == 'float64' :
            if  column != 'Failure Types':
                data_frame[column] = data_frame.groupby('Failure Types')[column].transform(lambda x: x.fillna(x.mean()))
        elif data_frame[column].dtype == 'object' :
            data_frame.dropna(subset=[column], inplace=True)

label_encoder = LabelEncoder()
for column in data_frame.select_dtypes(include='object').columns:
    data_frame[column] = label_encoder.fit_transform(data_frame[column])

# b2
numeric_cols = ['Air Temp (°C)', 'Process Temp (°C)', 'Rotational Speed (RPM)', 'Torque (Nm)', 'Tool Wear (Seconds)']

scaler = StandardScaler()
data_frame[numeric_cols] = scaler.fit_transform(data_frame[numeric_cols])


# c
#c1
data_frame['Failure_Binary'] = data_frame['Failure Types'].apply(lambda x: 'Failure' if x != 0 else 'No Failure')
print(data_frame['Failure_Binary'].value_counts())

#c2
sns.countplot(data=data_frame, x='Failure_Binary')
plt.title('Distribution of Tool Condition (Failure vs No Failure)')
plt.xlabel('Tool Condition')
plt.ylabel('Count')
plt.show()

#c4
#
X = data_frame.drop(columns=['Failure Types', 'Failure_Binary'])
y = data_frame['Failure_Binary']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y) 

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

print("After SMOTE:")
uniques, counts = unique(y_resampled, return_counts=True)
print(dict(zip(label_encoder.inverse_transform(uniques), counts)))

#c5

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

#c6

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM Linear": SVC(kernel='linear'),
    "SVM RBF": SVC(kernel='rbf')
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n{name}:\n")
    print("Confusion Matrix:\n", conf_matrix)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision (Failure)": report['1']['precision'],
        "Recall (Failure)": report['1']['recall'],
        "F1-Score (Failure)": report['1']['f1-score']
    })

#c7
# Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],  
}

grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_train, y_train)

print("Best Params (LR):", grid_lr.best_params_)

#KNN 
param_grid_knn = {'n_neighbors': range(1, 21)}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_knn.fit(X_train, y_train)
print("Best K for KNN:", grid_knn.best_params_)

#SVM linear
param_grid_svm_linear = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear']
}

grid_svm_linear = GridSearchCV(SVC(), param_grid_svm_linear, cv=5, scoring='accuracy')
grid_svm_linear.fit(X_train, y_train)

print("Best parameters for SVM (Linear):", grid_svm_linear.best_params_)

#SVM RBF
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf']
}
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
grid_svm.fit(X_train, y_train)
print("Best parameters for SVM (RBF):", grid_svm.best_params_)


#c8

comparison_table = DataFrame(results)
print(comparison_table.sort_values(by="Accuracy", ascending=False))

#d
#d1
X_multi = data_frame.drop(columns=['Failure Types', 'Failure_Binary'])
y_multi = data_frame['Failure Types']


smote_multi = SMOTE(random_state=42)
X_res_multi, y_res_multi = smote_multi.fit_resample(X_multi, y_multi)


X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_res_multi, y_res_multi, test_size=0.2, random_state=42)


models_multiclass = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM (One-vs-Rest)": OneVsRestClassifier(SVC(kernel='rbf')),
    "SVM (One-vs-one)":OneVsOneClassifier(SVC(kernel='linear', C=1))

}


results_multiclass = []

for name, model in models_multiclass.items():
    model.fit(X_train_m, y_train_m)
    y_pred_m = model.predict(X_test_m)

    acc = accuracy_score(y_test_m, y_pred_m)
    conf_matrix = confusion_matrix(y_test_m, y_pred_m)
    report = classification_report(y_test_m, y_pred_m, output_dict=True)

    print(f"\n{name}:\n")
    print("Confusion Matrix:\n", conf_matrix)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test_m, y_pred_m))

    results_multiclass.append({
        "Model": name,
        "Accuracy": acc,
        "Macro Precision": report['macro avg']['precision'],
        "Macro Recall": report['macro avg']['recall'],
        "Macro F1": report['macro avg']['f1-score']
    })

#d2
multi_comparison = DataFrame(results_multiclass)
print(multi_comparison.sort_values(by="Accuracy", ascending=False))

#d3
# KNN
param_grid_knn = {'n_neighbors': range(1, 21)}
grid_knn_multi = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_knn_multi.fit(X_train_m, y_train_m)
print("Best K for KNN (Multiclass):", grid_knn_multi.best_params_)

# Decision Tree
param_grid_dt = {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5)
grid_dt.fit(X_train_m, y_train_m)
print("Best params for Decision Tree:", grid_dt.best_params_)

# Random Forest
param_grid_rf = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
grid_rf.fit(X_train_m, y_train_m)
print("Best params for Random Forest:", grid_rf.best_params_)

# SVM One-vs-Rest
param_grid_svm_multi = {
    'estimator__C': [0.1, 1, 10],
    'estimator__gamma': [1, 0.1, 0.01]
}
grid_svm_multi = GridSearchCV(OneVsRestClassifier(SVC(kernel='rbf')), param_grid_svm_multi, cv=5)
grid_svm_multi.fit(X_train_m, y_train_m)
print("Best params for SVM (One-vs-Rest):", grid_svm_multi.best_params_)

# One-vs-One SVM
param_grid_svm_ovo = {
    'estimator__C': [0.1, 1, 10],
    'estimator__gamma': [1, 0.1, 0.01]
}
grid_svm_ovo = GridSearchCV(OneVsOneClassifier(SVC(kernel='rbf')),param_grid_svm_ovo,cv=5,scoring='accuracy')
grid_svm_ovo.fit(X_train_m, y_train_m)
print("Best params for SVM (One-vs-One):", grid_svm_ovo.best_params_)

#d4
comparison_multiclass = DataFrame(results_multiclass)
print(comparison_multiclass.sort_values(by="Accuracy", ascending=False))

