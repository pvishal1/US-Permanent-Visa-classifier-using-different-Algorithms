import numpy as np
import pandas as pd
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn import preprocessing, svm

import matplotlib.pyplot as plt
import seaborn as sns

'''
This program is implemented to predict possible decisions will be made for permanent US visa applications.

Authors: Pragya Vishalakshi & Burak Sivrikaya
'''


def classify():
    run_logreg = False
    run_tree = True
    run_forest = False
    run_adaboost = False
    run_knn = False
    run_mlp = False
    run_svm = False
    run_svm_bagging = False

    logreg_model = pd.Series()
    tree_model = pd.Series()
    forest_model = pd.Series()
    adaboost_model = pd.Series()
    knn_model = pd.Series()
    mlp_model = pd.Series()
    svm_bagging_model = pd.Series()

    # DATA RETRIEVAL & PRE-PROCESSING #

    # Read input data:
    raw_df = pd.read_csv('D:\\Data\\US_Permanent_Visas\\us_perm_visas.csv', header=0, low_memory=False)

    # Initial removal of some rows and columns:
    df = raw_df

    # Columns below were removed by intuition thinking they provide no information.
    df = df.drop(['employer_address_1', 'employer_postal_code', 'pw_soc_code', 'pw_source_name_9089'], 1)

    df = df.dropna(axis=1, thresh=300000)  # remove a row if it has any column with null values.
    df = df.dropna(axis=0, how='any')  # remove a row if it has any column with null values.

    # Data Transformation and feature selection were applied below:
    # Convert everything to year

    # Convert wage amount field to int64
    df['pw_amount_9089'] = df['pw_amount_9089'].str.replace(",", "")
    df['pw_amount_9089'] = np.floor(df['pw_amount_9089'].astype(float)).astype(np.int64)

    column_name = 'pw_amount_9089'
    mask = df.pw_unit_of_pay_9089 == "Hour"
    df.loc[mask, column_name] = df.loc[mask, column_name]*52*40

    mask = df.pw_unit_of_pay_9089 == "Week"
    df.loc[mask, column_name] = df.loc[mask, column_name]*52

    mask = df.pw_unit_of_pay_9089 == "Month"
    df.loc[mask, column_name] = df.loc[mask, column_name]*12

    mask = df.pw_unit_of_pay_9089 == "Bi-Weekly"
    df.loc[mask, column_name] = df.loc[mask, column_name]*26

    # Remove the amount that's greater than 7 digit
    df = df.drop(df[df["pw_amount_9089"] > 10000000].index)

    dict = {"Electronics Engineers":"Electronics Engineers","Teacher":"Education","Librarian":"Education","Manager":"Manager",
            "Dentist":"Medical","Physicians":"Medical","Pharmacists":"Medical","Medical":"Medical",
            "Software":"Software Developer","Programmer":"Software Developer","Developer":"Software Developer","Computer":"Software Developer","Database":"Software Developer",}

    categorize_titles_in_dict(df, dict)

    # Convert decision_date to decision_year
    df['decision_year'] = df['decision_date'].str[:4]
    df.decision_year = df.decision_year.astype(np.int64)
    df = df.drop(['decision_date'], 1)

    # Convert cerfitied-expired to certified
    df.loc[df['case_status'] == 'Certified-Expired', 'case_status'] = 'Certified'

    # Remove all withdrawn applications
    df = df[df.case_status != 'Withdrawn']

    # Removal of features according to correlation matrix and transformations
    df = df.drop(['employer_city', 'job_info_work_city', 'job_info_work_state', 'pw_unit_of_pay_9089'], 1)

    # Get the count of value_count on pw_soc_title <= 50
    # Convert those job titles to Other
    counts = df['pw_soc_title'].value_counts()
    mask = df['pw_soc_title'].isin(counts[counts <= 50].index)
    df.loc[mask, 'pw_soc_title'] = 'Other'

    # Get the count of value_count on employer_name <= 5
    # Convert those employer names to Other
    counts = df['employer_name'].value_counts()
    mask = df['employer_name'].isin(counts[counts <= 5].index)
    df.loc[mask, 'employer_name'] = 'Other'

    numerical_columns = ['pw_amount_9089', 'decision_year']
    numerical_data = df[numerical_columns]

    categorical_data = df
    categorical_data = categorical_data.drop(numerical_columns, 1)

    # Converting categorical variables to numerical variables by label encoding method:
    from collections import defaultdict
    d = defaultdict(preprocessing.LabelEncoder)

    # Encoding the variable
    categorical_data = categorical_data.apply(lambda x: d[x.name].fit_transform(x))

    # Inverse the encoded
    # fit = fit.apply(lambda x: d[x.name].inverse_transform(x))

    data = pd.concat([categorical_data, numerical_data], axis=1)
    # IMPORTANT: From now on, we will use data object.

    # Scaling values
    data_X = data.drop(['case_status'], 1)
    data_Y = data['case_status']
    scaler = preprocessing.MinMaxScaler()
    data_X[data_X.columns] = scaler.fit_transform(data_X[data_X.columns]) # Scale independent variables

    X = np.array(data_X)
    y = np.array(data_Y)

    # Correlation matrix
    # %config InlineBackend.figure_format = 'svg'
    # corrmat = data.corr()
    # f, ax = plt.subplots(figsize=(12, 9))
    # sns.heatmap(corrmat, vmax=.8, square=True, annot = True, fmt='.2f', annot_kws={'size': 8});
    # plt.show()

    # CLASSIFICATION MODELS #

    # Training - Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=17)

    # 5 fold stratified cross validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

    # Logistic Regression
    if run_logreg:
        print('Applying Logistic Regression...')
        logreg_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'random_state': [17]}
        logreg_model = fit_algorithm(LogisticRegression(), X_train, X_test, y_train, y_test, logreg_params, kf)

    # Decision Trees
    if run_tree:
        print('Applying Decision Tree...')
        tree_params = {'max_depth': [6, 7, 8], 'max_features': [8], 'random_state': [17]}
        tree_model = fit_algorithm(DecisionTreeClassifier(), X_train, X_test, y_train, y_test, tree_params, kf)

    # Random Forest
    if run_forest:
        print('Applying Random Forest...')
        forest_params = {'n_estimators': [50],
                         'max_depth': [6, 7, 8], 'max_features': [8], 'random_state': [17]}
        forest_model = fit_algorithm(RandomForestClassifier(), X_train, X_test, y_train, y_test, forest_params, kf)

    # Ada Boost
    if run_adaboost:
        print('Applying Ada Boost...')
        adaboost_params = {'n_estimators': [50], 'learning_rate': [1], 'random_state': [17]}
        adaboost_model = fit_algorithm(AdaBoostClassifier(base_estimator=DecisionTreeClassifier()), X_train, X_test, y_train, y_test, adaboost_params, kf)

    # kNN
    if run_knn:
        print('Applying k-Nearest Neighbor...')
        knn_params = {'n_neighbors': np.arange(6, 7).tolist()}
        knn_model = fit_algorithm(KNeighborsClassifier(), X_train, X_test, y_train, y_test, knn_params, kf)

    # SVM (Warning: Takes so much time!)
    if run_svm:
        print('Applying SVM...')
        svm_params = {'C': [0.001, 0.01, 0.1, 1, 10] , 'gamma': [0.001, 0.01, 0.1, 1],
                      'kernel': ['rbf', 'poly'], 'degree': [2, 3], #degree parameter ignored for kernels other than polynomial.
                      'class_weight': ['balanced']}
        svm_model = fit_algorithm(svm.SVC(probability=True), X_train, X_test, y_train, y_test, svm_params, kf)

    # Alternative: SVM with bagging classifier CV
    if run_svm_bagging:
        print('Applying SVM with BaggingClassifier')
        svm_bagging_params = {'max_samples': [5000] , 'n_estimators': [20]}
        svm_bagging_model = fit_algorithm(BaggingClassifier(svm.SVC(kernel='poly', C= 0.001, gamma=0.01, degree = 3, probability=True, class_weight='balanced')),
                                  X_train, X_test, y_train, y_test, svm_bagging_params, kf)

    # Multilayer Perceptron
    if run_mlp:
        mlp_params = {'solver': ['adam'] , 'alpha': [0.001],
                      'hidden_layer_sizes': [(5, 3), (4,2), (5,2)], 'activation': ['tanh'], 'random_state' : [17]}
        mlp_model = fit_algorithm(MLPClassifier(), X_train, X_test, y_train, y_test, mlp_params, kf)

    # Data frame made of results
    summary = pd.concat([logreg_model, tree_model, forest_model, adaboost_model, knn_model, mlp_model, svm_bagging_model], axis=1)
    summary.columns = ['Logistic Regression', 'Decision Trees', 'Random Forest', 'KNN', 'Adaboost', 'MLP', 'SVM']
    print(summary)

    # Lets create best model again (which was random forest)
    final_forest = RandomForestClassifier(n_estimators=50, max_depth=8, max_features=8, random_state=17, n_jobs=-1)
    final_forest.fit(X_train, y_train)
    importances = final_forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in final_forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot the feature importance of the forest
    plt.figure(figsize=(10, 8))
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices], color="g", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), data.columns[1:9], rotation = 90)
    plt.show()


# Categorizing soc_title column
def categorize_titles_in_dict(df, dict):
    for job_title, category in dict.items():
        categorize_job_title(df, job_title, category)


def categorize_job_title(df, job_title, category):
    # masking other titles to find and replace
    mask = (df.pw_soc_title.str.contains(job_title))
    df.loc[mask, 'pw_soc_title'] = category


# Defining fit_algorithm function
def fit_algorithm(alg, X_train, X_test, y_train, y_test, parameters, cv=5):
    """
    This function will split our dataset into training and testing subsets, fit cross-validated
    GridSearch object, test it on the holdout set and return some statistics
    """
    grid = GridSearchCV(alg, parameters, cv=cv, n_jobs=-1, scoring='roc_auc')
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    y_pred_prob = grid.predict_proba(X_test)
    y_pred_pos_prob = [p[1] for p in y_pred_prob] # Get the positive class
    confmat = confusion_matrix(y_test,y_pred)

    return pd.Series({
        "train_auc_roc": np.around(grid.best_score_, decimals=3).astype(str),
        "test_acc": np.around(accuracy_score(y_test, y_pred), decimals=3).astype(str),
        "auc_roc" : np.around(roc_auc_score(y_test, y_pred_pos_prob), decimals=3).astype(str),
        "p": np.around(precision_score(y_pred, y_test), decimals=3).astype(str),
        "r": np.around(recall_score(y_pred, y_test),decimals=3).astype(str),
        "f1": np.around(f1_score(y_pred, y_test),decimals=3).astype(str),
        "best_params": [grid.best_params_],
        "true_negatives": confmat[0,0],
        "false_negatives": confmat[1,0],
        "true_positives": confmat[1,1],
        "false_positives": confmat[0,1]
        })

if __name__ == "__main__":
    classify()
