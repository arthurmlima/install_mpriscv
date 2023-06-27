"""
Criado por: Andre Luiz Santos e Ivan C Perissini
Data: 22/07/2020
Função: Módulo com conjunto usual de ferramentas para trabalhar com aprendizagem de máquina
Última alteração: 24/07/2020
"""
from core.demeter import parameters
from core.demeter import descriptors as info
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
from datetime import datetime
from collections import Counter
import time
import re
from joblib import dump, load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import export_graphviz
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import pydot
import os
import imblearn


def __version__():
    return 'mltools version: v0.2'


# ======================== SUPPORT =========================

# ~~~~~~~~~~~~~~~~~~ Data preparation~~~~~~~~~~~~~~~~~~~
# Given a pandas data frame the function removes or keeps the provided column names
# The function also separates the data into input(X) and output(Y), while also separating the test and train data
# >> Sample 1: remove_list = ['hist', 'nu', 'hu']
# >> Sample 2: keep_only_list = ['Mean', 'Max', 'Min']
def prepare_data(db_data, y_name='Y', test_size=0.4, random_state=20, keep_only_list=[], remove_list=[]):
    column_size = db_data.shape[1]

    # Keep only the selected list columns, remove all others
    if len(keep_only_list) > 0:
        keep_only_list.append(y_name)
        db_data = filter_col_data(db_data, column_list=keep_only_list, remove=False)

    # Remove all columns based on remove list input
    if len(remove_list) > 0:
        db_data = filter_col_data(db_data, column_list=remove_list, remove=True)

    # Separate the classes from descriptors
    y = db_data[y_name].values
    X = db_data.drop(labels=[y_name], axis=1)

    # Split data for train and test
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = X, (), y, ()

    return X_train, X_test, y_train, y_test


# ~~~~~~~~~~~~~~~~~~ Filter col data~~~~~~~~~~~~~~~~~~~
# Function to reduce the data frame based on a column name
def filter_col_data(db_data, column_list=[], remove=True):
    column_size = db_data.shape[1]

    if remove:
        for column in column_list:
            db_data = db_data.loc[:, ~db_data.columns.str.contains(column)]
            print('Removing based on ', column, ':', column_size - db_data.shape[1], 'columns removed')
            column_size = db_data.shape[1]
    else:
        keep_string = '|'.join(column_list)
        db_data = db_data.filter(regex=keep_string, axis=1)
        print('Keeping only columns with', keep_string, ':', column_size - db_data.shape[1], 'columns removed')

    return db_data


# ~~~~~~~~~~~~~~~~~~ Filter row data~~~~~~~~~~~~~~~~~~~
# Function to reduce the data frame based on a column name and an interval
def filter_numerical_row_data(db_data, column_name='GPS_alt', lower_limit=-np.inf, upper_limit=np.inf, remove=False):
    row_size = db_data.shape[0]
    if remove:
        db_data = db_data.loc[~((db_data[column_name] >= lower_limit) & (db_data[column_name] <= upper_limit))]
        print('Removing based on', column_name, 'at given interval: ', row_size - db_data.shape[0], 'lines removed')
    else:
        db_data = db_data.loc[(db_data[column_name] >= lower_limit) & (db_data[column_name] <= upper_limit)]
        print('Keeping based on', column_name, 'at given interval: ', row_size - db_data.shape[0], 'lines removed')

    return db_data


# ~~~~~~~~~~~~~~~~~~ Filter row data~~~~~~~~~~~~~~~~~~~
# Function to reduce the data frame based on a column name and row name
def filter_categorical_row_data(db_data, column_name='season', value='water', remove=False):
    row_size = db_data.shape[0]
    if remove:
        db_data = db_data.loc[db_data[column_name] != value]
        print('Removing', value, 'from', column_name, ':', row_size - db_data.shape[0], 'lines removed')
    else:
        # db_data = db_data.loc[db_data[column_name] == value]
        db_data = db_data.loc[db_data[column_name] == value]
        print('Keeping only', value, 'on', column_name, ':', row_size - db_data.shape[0], 'lines removed')

    return db_data


# ~~~~~~~~~~~~~~~~~~ PCA ~~~~~~~~~~~~~~~~~~~
# Uses Principal Component Analysis on a X and y set, into k given components
# This is mainly used to reduce the data set dimension and facilitate calculations and learning process
# Can also be used for data visualization
def pca_reduction(X, y, k=5):
    pca = PCA(n_components=k)
    principal_components = pca.fit_transform(X)
    principal_df = pd.DataFrame(principal_components)
    y = pd.Series(data=y, name='label')
    pca_df = pd.concat([principal_df, y], axis=1)

    pca_ratio = pca.explained_variance_ratio_
    pca_names = ['PC' + str(n + 1) for n in range(k)]

    print('PCA explained variance ratio:')
    print(pca_ratio)

    print('PCA explained variance Sum:')
    print(np.sum(pca_ratio))

    fig, ax = plt.subplots()
    ax.bar(pca_names, pca_ratio)
    # ax.set_xticks(np.arange(pca_names.shape[0]))
    ax.set_xticklabels(pca_names, rotation=60, ha='right')
    fig.tight_layout()
    plt.show()

    return pca_df


# ======================== DATABASE ANALYSIS =========================

# ~~~~~~~~~~~~~~~~~~ Category Evaluation ~~~~~~~~~~~~~~~~~~~
def category_evaluation(data, show_results=False):
    category_counter = Counter(data)
    results = {}

    # Sort by name
    # sort_category = category_counter
    sort_category = {key: value for key, value in sorted(category_counter.items(),
                                                         key=lambda item: item[0],
                                                         reverse=False)}
    # Sort by quantity
    sort_values = {key: value for key, value in sorted(category_counter.items(),
                                                       key=lambda item: item[1],
                                                       reverse=True)}

    # Balance Metrics
    n_data = results['category_n_data'] = len(data)
    n_category = results['category_n_categories'] = len(sort_category)

    count_vet = [float(count) for category, count in sort_category.items()]
    results['category_Mean'] = np.mean(count_vet)
    results['result_ImbalanceRatio'] = (np.max(count_vet) / np.min(count_vet))
    # Shannon entropy / SHANNON DIVERSITY INDEX (LET)
    H = -sum([(count / n_data) * np.log((count / n_data)) for count in count_vet])
    if n_category > 1:
        results['result_BalanceIndex'] = H / np.log(n_category)
    else:
        results['result_BalanceIndex'] = 1

    if show_results:
        vet_category = np.array(list(sort_category.items()))
        category_names = vet_category[:, 0].astype(np.str)
        category_count = vet_category[:, 1].astype(np.int)

        vet_sort_category = np.array(list(sort_values.items()))
        category_sort_names = vet_sort_category[:, 0].astype(np.str)
        category_sort_count = vet_sort_category[:, 1].astype(np.int)

        plt.close('all')
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('Category Balance Index: ' + f"{results['result_BalanceIndex']:.3f}"
                     + "\n Category Imbalance Ratio: " + f"{results['result_ImbalanceRatio']:.3f}", fontsize=16)

        ax1.bar(category_names, category_count)
        ax1.set_title('Distribution')
        ax1.set_xticks(np.arange(category_names.shape[0]))
        ax1.set_xticklabels(category_names, rotation=60, ha='right')

        ax2.bar(category_sort_names, category_sort_count)
        ax2.set_title('Rank')
        ax2.set_xticks(np.arange(category_sort_names.shape[0]))
        ax2.set_xticklabels(category_sort_names, rotation=60, ha='right')

        results_filter = {key: value for key, value in sort_values.items() if value / n_data > 0.01}
        wedges, texts, texts = ax3.pie(results_filter.values(), labels=results_filter.keys(), autopct='%1.1f%%')
        plt.setp(texts, size=10, weight="bold")

        plt.show()

    results.update({('count_' + str(key)): value for key, value in sort_category.items()})

    summary_category = {key: value for key, value in results.items() if 'result' in key}
    print('Category Evaluation Summary:', summary_category)

    return results, sort_category


# ~~~~~~~~~~~~~~~~~~ Category Evaluation ~~~~~~~~~~~~~~~~~~~
# ---sampling_strategy---
# string -> 'auto' / 'all' / 'majority' / 'minority' / 'not majority' / 'not minority'
# float -> 0.8 (only for binary classification)
# dictionary -> {label1: val1, label2: val2, label3: val3}
# ---over sampling method---
# imblearn.over_sampling.SMOTE()  # Synthetic Minority Oversampling TEchnique
# imblearn.over_sampling.SVMSMOTE()  # Generates samples away from the region class overlaps
# imblearn.over_sampling.BorderlineSMOTE()  # Generates samples towards the region class overlaps
# imblearn.over_sampling.ADASYN() # Generates samples inversely proportional to the density of the

def data_base_balance(X, y, method='auto', sampling_strategy=''):
    steps = []

    # ----Automatic ideal class dictionary generation----
    if sampling_strategy == 'mean':
        original_class = Counter(y)
        mean_ref = int(np.mean(list(original_class.values())))
        sampling_strategy = {key: mean_ref for key, value in original_class.items()}

    if sampling_strategy == 'moderate':
        rate = 5
        original_class = Counter(y)
        mean_ref = int(np.mean(list(original_class.values())))
        sampling_strategy = {}
        for key, value in original_class.items():
            if (mean_ref/value) > rate:
                sampling_strategy[key] = int(rate * value)
            elif (mean_ref/value) < 1/rate:
                sampling_strategy[key] = int((1/rate) * value)
            else:
                sampling_strategy[key] = int(mean_ref)

    elif 'int' in str(type(sampling_strategy)):
        original_class = Counter(y)
        ref = sampling_strategy
        sampling_strategy = {key: ref for key, value in original_class.items()}

    # ----Ideal Dictionary matching----
    # If a dictionary is given or automatic generated it concatenates over and under sampling strategies
    if 'dict' in str(type(sampling_strategy)):
        ideal_class = sampling_strategy

        original_class = Counter(y)
        original_class = {key: original_class[key] for key, value in sampling_strategy.items()}

        over_sampling = {}
        over_sampling.update(original_class)
        over_sampling.update({key: value for key, value in ideal_class.items() if value >= original_class[key]})

        under_sampling = {}
        under_sampling.update(over_sampling)
        under_sampling.update({key: value for key, value in ideal_class.items() if value <= original_class[key]})
    else:
        over_sampling = under_sampling = sampling_strategy

    # ----Over and Undersample strategy parametrization----
    if method == 'over' or method == 'auto':
        # print('over', over_sampling)
        over_sample = imblearn.over_sampling.SMOTE(sampling_strategy=over_sampling)
        steps.append(('over', over_sample))

    if method == 'under' or method == 'auto':
        # print('under', under_sampling)
        under_sample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=under_sampling)
        steps.append(('under', under_sample))

    # ----Pipeline generation and actual data re-sampling----
    pipeline = imblearn.pipeline.Pipeline(steps=steps)
    X_balance, y_balance = pipeline.fit_resample(X, y)
    print('The Data was Re-sampled using the following steps:')
    print(steps)

    return X_balance, y_balance


# ======================== FEATURE ANALYSIS =========================

# ~~~~~~~~~~~~~~~~~~ Feature variation ~~~~~~~~~~~~~~~~~~~
# If a threshold is provided all columns with variations below the given value are removed from the data frame
# Also computes the descriptors variation, and highlights the k greatest and lowest variation values
# Promotes feature selection by removing low variation descriptors
def feature_variation(X, k=5, threshold=-1):
    low_var_df = X

    if threshold >= 0:
        selection = VarianceThreshold(threshold=threshold)
        selection.fit_transform(X)
        low_var_df = X[X.columns[selection.get_support(indices=True)]]

    feat_var = low_var_df.var()
    feat_var.sort_values(inplace=True, ascending=False)
    col_names = np.array(list(feat_var.keys()))
    vet_var_feature = np.array(list(feat_var.values))

    fig = plt.figure()

    ax0 = fig.add_subplot(3, 1, 1)
    ax0.set_title("All Features Variance")
    ax0.bar(col_names, vet_var_feature)
    ax0.get_xaxis().set_visible(False)

    # Selects the k highest values
    col_names_filter = col_names[:k]
    vet_var_feature_filter = vet_var_feature[:k]

    ax1 = fig.add_subplot(3, 2, 3)
    ax1.set_title("Highest Features Variance")
    ax1.bar(col_names_filter, vet_var_feature_filter)
    ax1.set_xticks(np.arange(col_names_filter.shape[0]))
    ax1.set_xticklabels(col_names_filter, rotation=60, ha='right')

    # Selects the k lowest values
    col_names_filter = col_names[-k:]
    vet_var_feature_filter = vet_var_feature[-k:]

    ax2 = fig.add_subplot(3, 2, 4)
    ax2.set_title("Lowest Features Variance")
    ax2.bar(col_names_filter, vet_var_feature_filter)
    ax2.set_xticks(np.arange(col_names_filter.shape[0]))
    ax2.set_xticklabels(col_names_filter, rotation=60, ha='right', rotation_mode="anchor")
    plt.show()

    return low_var_df, feat_var


# ~~~~~~~~~~~~~~~~~~ Feature Analysis Categorical ~~~~~~~~~~~~~~~~~~~
# A set of features analysis tools for categorical output (y) where the function return the k most relevant features
# Chi square - chi2 - Input data can be categorical, if not the data is converted into bins
# Extra Trees Classifier - tree - Uses the most relevant tree nodes to select the features
# Random Forest Classifier - RF - Uses the most relevant tree nodes to select the features
def feature_analysis_categorical(X, y, k=5, method='chi2'):
    # Feature extraction
    if method == 'chi2':
        method_name = 'Chi square'
        selection = SelectKBest(score_func=chi2, k=k)
    elif method == 'tree':
        method_name = 'Extra Trees Classifier'
        selection = ExtraTreesClassifier(n_estimators=10, random_state=0)
    elif method == 'RF':
        method_name = 'Random Forest Classifier'
        selection = RandomForestClassifier(n_estimators=10, random_state=0)
    else:
        print('Error: Invalid method', method)
        print('Current methods available are: [chi2 , tree, RF]')
        return -1

    fit = selection.fit(X, y)
    col_names = list(X.columns)
    feature = {}

    if method == 'tree' or method == 'RF':
        for index, name in enumerate(col_names):
            feature[name] = selection.feature_importances_[index]
    else:
        for index, name in enumerate(col_names):
            feature[name] = fit.scores_[index]

    sort_feature = {key: value for key, value in sorted(feature.items(), key=lambda item: item[1], reverse=True)}

    sort_vet_feature = np.array(list(sort_feature.items()))
    feature_names = sort_vet_feature[:k, 0]
    feature_importance = sort_vet_feature[:k, 1]
    feature_importance = feature_importance.astype(np.float)

    fig, ax = plt.subplots()
    ax.bar(feature_names, feature_importance)
    ax.set_title("Feature Importance - " + method_name)
    ax.set_xticks(np.arange(feature_names.shape[0]))
    ax.set_xticklabels(feature_names, rotation=60, ha='right')
    fig.tight_layout()
    plt.show()

    return sort_feature


# ~~~~~~~~~~~~~~~~~~ Feature Analysis Quantitative ~~~~~~~~~~~~~~~~~~~
# A set of features analysis tools for quantitative output (y) where the function return the k most relevant features
# Univariate linear regression test - f_regression - Linear model for testing the individual effect of each feature
# ANOVA F-value - f_classif - Recommended when the input (X) is categorical and the output (y) is quantitative
# Mutual info regression - mir - Nonparametric methods based on entropy estimation from k-nearest neighbors distances
def feature_analysis_quantitative(X, y, k=5, method='f_regression', random_state=20):
    # Feature extraction
    if method == 'f_regression':
        method_name = 'Univariate linear regression test'
        selection = SelectKBest(score_func=f_regression, k=k)
    elif method == 'f_classif':
        method_name = 'ANOVA F-value'
        selection = SelectKBest(score_func=f_classif, k=k)
    elif method == 'mir':
        method_name = 'Mutual info regression'
        selection = SelectKBest(score_func=mutual_info_regression, k=k)
    elif method == 'tree':
        method_name = 'Random Forest Regressor'
        selection = RandomForestRegressor(max_depth=k, random_state=random_state)
    else:
        print('Error: Invalid method', method)
        print('Current methods available are: [f_regression , f_classif, mir, tree]')
        return -1

    fit = selection.fit(X, y)

    col_names = list(X.columns)
    feature = {}

    if method == 'tree':
        prediction = selection.predict(X)
        score = metrics.r2_score(y, prediction)
        title_text = ("Feature Importance - " + method_name +
                      '\n Coefficient of determination score: ' + f"{score:.4f}")
        for index, name in enumerate(col_names):
            feature[name] = selection.feature_importances_[index]
    else:
        title_text = "Feature Importance - " + method_name
        for index, name in enumerate(col_names):
            feature[name] = fit.scores_[index]

    sort_feature = {key: value for key, value in sorted(feature.items(), key=lambda item: item[1], reverse=True)}

    sort_vet_feature = np.array(list(sort_feature.items()))
    feature_names = sort_vet_feature[:k, 0]
    feature_importance = sort_vet_feature[:k, 1]
    feature_importance = feature_importance.astype(np.float)

    fig, ax = plt.subplots()
    ax.bar(feature_names, feature_importance)
    ax.set_title(title_text)
    ax.set_xticks(np.arange(feature_names.shape[0]))
    ax.set_xticklabels(feature_names, rotation=60, ha='right')
    fig.tight_layout()
    plt.show()

    return sort_feature, selection


# ~~~~~~~~~~~~~~~~~~ Feature Analysis Regularization ~~~~~~~~~~~~~~~~~~~
# A form of regression, that constrains/ regularizes or shrinks the coefficient estimates towards zero
# Regularization, significantly reduces the variance of the model, without substantial increase in its bias
# >> L1 regularization or LASSO regression:
# Parameters are reduced to zero to features that don't effect the cost function
# In lasso, one of the correlated predictors has a larger coefficient, while the rest are (nearly) zeroed
# Better when only a few predictors actually influence the response
# >> L2 regularization or Ridge regression:
# Prevent over fitting in the training data by reducing the variance
# In ridge regression, the coefficients of correlated predictors are similar;
# Better when most predictors impact the response
# >> ElasticNet:
# Simply a convex combination of Ridge and Lasso
def feature_analysis_regularization(X, y, k=5, alpha=1.0, method='L2', ratio=0.5):
    if method == 'L1':
        regularization = Lasso(alpha=alpha, max_inter=20000)
    elif method == 'L2':
        regularization = Ridge(alpha=alpha, max_inter=20000)
    elif method == 'ElasticNet':
        regularization = ElasticNet(alpha=alpha, l1_ratio=ratio, random_state=0, max_inter=20000)
    else:
        print('Error: Invalid method', method)
        print('Current available are: [L1 , L2, ElasticNet]')
        return -1

    regularization.fit(X, y)
    score = regularization.score(X, y)

    col_names = list(X.columns)
    feature = {}
    for index, name in enumerate(col_names):
        feature[name] = regularization.coef_[index]

    sort_feature = {key: value for key, value in sorted(feature.items(), key=lambda item: abs(item[1]), reverse=True)}

    sort_vet_feature = np.array(list(sort_feature.items()))
    feature_names = sort_vet_feature[:k, 0]
    feature_rank = sort_vet_feature[:k, 1]
    feature_rank = feature_rank.astype(np.float)

    fig, ax = plt.subplots()
    ax.bar(feature_names, feature_rank)
    ax.set_title('Feature Coefficient - ' + method + ' regularization with alpha = ' + str(alpha)
                 + '\n Coefficient of determination score: ' + f"{score:.4f}")
    ax.set_xticks(np.arange(feature_names.shape[0]))
    ax.set_xticklabels(feature_names, rotation=60, ha='right')
    fig.tight_layout()
    plt.show()

    return sort_feature, regularization


# ~~~~~~~~~~~~~~~~~~ Feature Analysis Regularization Set ~~~~~~~~~~~~~~~~~~~
# Simply applies and show the results of several alpha values for the selected method
def feature_analysis_regularization_set(X, y, k=5, method='L2', ratio=0.5):
    alpha_set = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    size = len(alpha_set)

    sort_vet_feature_set = np.empty((size, X.shape[1], 2), dtype=object)
    score_set = np.empty(size)

    for alpha_i, alpha in enumerate(alpha_set):
        if method == 'L1':
            regularization = Lasso(alpha=alpha)
        elif method == 'L2':
            regularization = Ridge(alpha=alpha)
        elif method == 'ElasticNet':
            regularization = ElasticNet(alpha=alpha, l1_ratio=ratio, random_state=0)
        else:
            print('Error: Invalid method', method)
            print('Current available are: [L1 , L2, ElasticNet]')
            return -1

        regularization.fit(X, y)
        score = regularization.score(X, y)
        score_set[alpha_i] = score

        col_names = list(X.columns)
        feature = {}
        for index, name in enumerate(col_names):
            feature[name] = regularization.coef_[index]

        sort_feature = {key: value for key, value in
                        sorted(feature.items(), key=lambda item: abs(item[1]), reverse=True)}
        sort_vet_feature = np.array(list(sort_feature.items()))
        sort_vet_feature_set[alpha_i] = sort_vet_feature

    fig, ax = plt.subplots()
    fig.suptitle('Feature Coefficient Analysis with ' + method, fontsize=16)
    ax.axis("off")

    for alpha_i, alpha in enumerate(alpha_set):

        feature_names = sort_vet_feature_set[alpha_i][:k, 0]
        feature_rank = sort_vet_feature_set[alpha_i][:k, 1]
        feature_rank = feature_rank.astype(np.float)

        row = 3
        col = 3

        if alpha_i > 2:
            p = alpha_i + 4
        else:
            p = alpha_i + 1

        ax = fig.add_subplot(row, col, p)
        ax.bar(feature_names, feature_rank)
        ax.set_title('Regularization with alpha = ' + str(alpha)
                     + '\n Coefficient of determination score: ' + f"{score_set[alpha_i]:.4f}")
        ax.set_xticks(np.arange(feature_names.shape[0]))
        ax.set_xticklabels(feature_names, rotation=60, ha='right')

    plt.show()

    return sort_vet_feature_set


# ~~~~~~~~~~~~~~~~~~ Recursive Feature Elimination ~~~~~~~~~~~~~~~~~~~
# Given an external estimator that assigns weights to features, the goal of recursive feature elimination (RFE)
# is to select features by recursively considering smaller and smaller sets of features.
# Each iteration a 'step' number of the least influential features are removed, if an 'step' between 0 and 1
# is provided, a percentage of the remaining features are considered.
# At the end the k best features are showed and returned as a reduced model
# >> Logistic Regression with lbfgs solver - 'lbfgs' - Requires categorial output data (y)
# >> Logistic Regression with saga solver - 'saga' - Requires categorial output data (y)
# >> Support Vector Regression with linear kernel - 'svr_l' - Requires quantitative output data (y)
def feature_elimination(X, y, k=5, method='lbfgs', step=1):
    # Feature extraction model
    if method == 'lbfgs':
        model = LogisticRegression(dual=False, max_iter=5000, solver='lbfgs')
    elif method == 'saga':
        model = LogisticRegression(dual=False, max_iter=5000, solver='saga')
    elif method == 'svr_l':
        model = SVR(kernel="linear")
    else:
        print('Error: Invalid method', method)
        print('Current methods available are: [lbfgs , saga, svr_l]')
        return -1

    rfe = RFE(model, n_features_to_select=k, step=step)
    fit = rfe.fit(X, y)
    score = rfe.score(X, y)
    # print("Num Features: %s" % fit.n_features_)
    # print('Selected Features:', list(X.columns[fit.support_]))

    col_names = list(X.columns)
    feature = {}
    for index, name in enumerate(col_names):
        feature[name] = fit.ranking_[index]

    sort_feature = {key: value for key, value in sorted(feature.items(), key=lambda item: item[1], reverse=False)}

    sort_vet_feature = np.array(list(sort_feature.items()))
    feature_names = sort_vet_feature[:2 * k, 0]
    feature_rank = sort_vet_feature[:2 * k, 1]
    feature_rank = feature_rank.astype(np.float)

    fig, ax = plt.subplots()
    ax.bar(feature_names, feature_rank)
    ax.set_title("Feature Rank by " + method + 'model' + '\n' + str(k) + ' features RFE score: ' + f"{score:.4f}")
    ax.set_xticks(np.arange(feature_names.shape[0]))
    ax.set_xticklabels(feature_names, rotation=60, ha='right')
    fig.tight_layout()
    plt.show()

    return sort_feature, rfe


# ~~~~~~~~~~~~~~~~~~ Descriptor evaluation ~~~~~~~~~~~~~~~~~~~
# Given a data frame, filters, and number of intended features several features methods are used in sequence
# Returning graphical and numerical information for each evaluation
def descriptor_evaluation(db_data, remove_descriptor=['nSP'], keep_descriptor=[],
                          y_name='label', y_type='', n_features=5, random_state=20, test_size=0.4):
    X_train, X_test, y_train, y_test = prepare_data(db_data,
                                                    y_name=y_name,
                                                    test_size=test_size,
                                                    random_state=random_state,
                                                    remove_list=remove_descriptor,
                                                    keep_only_list=keep_descriptor)

    print()
    print('\n~~~~~~~ DESCRIPTOR EVALUATION ~~~~~~~')

    # If no type is provided, determines the output type based on y first data
    if y_type == '':
        if 'str' in str(type(y_train[0])):
            y_type = 'categorical'
            print('Output data type was considered categorical')
        else:
            y_type = 'quantitative'
            print('Output data type was considered quantitative')

    print('Feature variation evaluation')
    low_var_df, var_list = feature_variation(X_train, k=n_features, threshold=0)

    # Feature importance calculations based on data type
    if y_type == 'categorical':
        print('Categorical Feature Analysis')
        feat = feature_analysis_categorical(X_train, y_train, k=n_features, method='chi2')
        feat = feature_analysis_categorical(X_train, y_train, k=n_features, method='tree')
        feat = feature_analysis_categorical(X_train, y_train, k=n_features, method='RF')

        print('Categorical Recursive Feature Elimination (RFE)')
        feat, rfe_model = feature_elimination(X_train, y_train, k=n_features, method='lbfgs', step=0.05)
        feat, rfe_model = feature_elimination(X_train, y_train, k=n_features, method='saga', step=0.05)
        if test_size > 0:
            print(n_features, ' features score:', rfe_model.score(X_test, y_test))

    elif y_type == 'quantitative':
        print('Quantitative Feature Analysis')
        feat, q_model = feature_analysis_quantitative(X_train, y_train, k=n_features, method='f_regression')
        feat, q_model = feature_analysis_quantitative(X_train, y_train, k=n_features, method='f_classif')
        feat, q_model = feature_analysis_quantitative(X_train, y_train, k=n_features, method='mir')
        feat, q_model = feature_analysis_quantitative(X_train, y_train, k=n_features, method='tree')

        print('Quantitative Recursive Feature Elimination (RFE)')
        feat, rfe_model = feature_elimination(X_train, y_train, k=n_features, method='svr_l', step=0.05)
        if test_size > 0:
            print(n_features, ' features score:', rfe_model.score(X_test, y_test))

        print('Regularization coefficient analysis')
        feature_analysis_regularization_set(X_train, y_train, k=n_features, method='L1', ratio=0.5)
        feature_analysis_regularization_set(X_train, y_train, k=n_features, method='L2', ratio=0.5)
        feature_analysis_regularization_set(X_train, y_train, k=n_features, method='ElasticNet', ratio=0.5)


# ======================== MODELING =========================

# ~~~~~~~~~~~~~~~~~~ Train model ~~~~~~~~~~~~~~~~~~~
# Function that given a model and data, will train and save the results in a joblib file for later usage
def train_model(model, X_train, y_train, model_name='', save_model=False):
    # Train model
    model.fit(X_train, y_train)
    print('Model trained')

    # Save results if requested
    if save_model:
        if model_name == '':
            model_name = 'model_' + datetime.now().strftime('%d_%m_%Y') + datetime.now().strftime('_H%H_%M')
        else:
            model_name = model_name + '_' + str(int(time.time()))
        dump(model, model_name + '.joblib')

    return model


# ======================== RESULTS =========================

# ~~~~~~~~~~~~~~~~~~ Model metrics ~~~~~~~~~~~~~~~~~~~
# Given a model and the test input and output data, the function show several results including a confusion matrix
def model_metrics(model, X, y, class_names=[], y_type='', show_matrix=True):
    print('\n~~~~~~~ MODEL METRICS ~~~~~~~')
    # If no type is provided, determines the output type based on y first data
    if y_type == '':
        if 'str' in str(type(y[0])):
            y_type = 'categorical'
            print('Output data type was considered categorical\n')
        else:
            y_type = 'quantitative'
            print('Output data type was considered quantitative\n')

    # Classification metrics
    if y_type == 'categorical':
        cmap_list = ['Greens', 'Blues', 'YlGn', 'binary', 'hot']
        cmap_option = cmap_list[0]

        prediction_test = model.predict(X)
        labels = np.unique(y)

        print('Total Accuracy:', metrics.accuracy_score(y, prediction_test), '\n')

        matrix = confusion_matrix(y, prediction_test, labels=labels)
        print(matrix, '\n')

        report = classification_report(y, prediction_test, labels=labels, output_dict=False)
        report_dic = classification_report(y, prediction_test, labels=labels, output_dict=True)

        print(report, '\n')

        # Get label names from external function and only the relevant data
        if not class_names:
            for key, value in parameters.get_label_info().items():
                if value[0] in labels:
                    # class_names.append(key)
                    class_names.append(value[-1])

        if show_matrix:
            plot_confusion_matrix(model, X, y,
                                  labels=labels,
                                  display_labels=class_names,
                                  cmap=cmap_option,
                                  normalize='true')

            plt.show()

        return report_dic

    # Regression metrics
    elif y_type == 'quantitative':
        prediction_test = model.predict(X)

        result_explained_variance_score = metrics.explained_variance_score(y, prediction_test)
        print('Explained variance score:', result_explained_variance_score)

        result_r2_score = metrics.r2_score(y, prediction_test)
        print('R2 score (coefficient of determination):', result_r2_score)

        result_max_error = metrics.max_error(y, prediction_test)
        print('Max Error:', result_max_error)

        result_mean_absolute_error = metrics.mean_absolute_error(y, prediction_test)
        print('Mean Absolute Error:', result_mean_absolute_error)

        residuals = y - prediction_test

        fig = plt.figure()

        # (Y x Prediction) Plot
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title('Results comparison')  # Nomeia cada subplot
        ax.set_xlabel("True y")
        ax.set_ylabel("Predicted y")
        ax.scatter(y, prediction_test)

        z = np.polyfit(y, prediction_test, 1)
        p = np.poly1d(z)
        ax.plot(y, p(y), "b-", lw=1)
        text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {result_r2_score:0.3f}$"
        ax.text(np.min(y), np.max(prediction_test) * 0.9, text)

        # Residual Plot
        ax = fig.add_subplot(1, 2, 2)
        ax.set_title('Residual evaluation')  # Nomeia cada subplot
        ax.set_xlabel("True y")
        ax.set_ylabel("Residual")
        ax.axhline(c='grey', lw=1)
        ax.scatter(y, residuals, color='black')

        plt.show()

    return 0


def cross_validation_metrics(model, X, y, cv=5):
    # If the estimator is a classifier and y is multiclass, StratifiedKFold is used
    scores1 = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    scores2 = cross_val_score(model, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    scores3 = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    scores4 = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
    scores5 = cross_val_score(model, X, y, cv=cv, scoring='roc_auc_ovr', n_jobs=-1)

    print('Cross validation results for a', cv, 'KFold analysis:')
    print('accuracy:', scores1)
    print('balanced_accuracy:', scores2)
    print('f1_macro:', scores3)
    print('f1_weighted:', scores4)
    print('roc_auc_ovr:', scores5)

    score_dic = {}
    score_dic.update({'accuracy_KFold_' + str(key + 1): val for key, val in enumerate(scores1)})
    score_dic.update({'balanced_accuracy_KFold_' + str(key + 1): val for key, val in enumerate(scores2)})
    score_dic.update({'f1_macro_KFold_' + str(key + 1): val for key, val in enumerate(scores3)})
    score_dic.update({'f1_weighted_KFold_' + str(key + 1): val for key, val in enumerate(scores4)})
    score_dic.update({'roc_auc_ovr_KFold_' + str(key + 1): val for key, val in enumerate(scores5)})

    return score_dic


# ~~~~~~~~~~~~~~~~~~ Model show tree ~~~~~~~~~~~~~~~~~~~
# Given a tree model and a input sample, the function show several results including a confusion matrix
def model_show_tree(tree_model, X_sample, class_names='', n_tree=1,
                    output_path=parameters.get_result_folder() + r'\\tree\\'):
    # Add Graphviz path to enviroments to allow dot file output usage
    # os.environ["PATH"] += os.pathsep + 'bin/Graphviz/bin/'
    os.environ["PATH"] += os.pathsep + parameters.get_graphviz_folder()

    # Get the column names from the input sample
    col_names = list(X_sample.columns)
    # Get label names from external function
    if class_names == '':
        class_names = [key for key, value in parameters.get_label_info().items()]

    for i in range(n_tree):
        n = np.random.randint(0, high=len(tree_model.estimators_))

        estimator = tree_model.estimators_[n]
        dot_path = output_path + 'tree_' + str(n) + '.dot'
        out_path = output_path + 'tree_' + str(n) + '.pdf'

        export_graphviz(estimator, out_file=dot_path,
                        feature_names=col_names,
                        class_names=class_names,
                        rounded=True, proportion=False,
                        precision=2, filled=True)

        # pydot
        (graph,) = pydot.graph_from_dot_file(dot_path)
        graph.write_pdf(out_path)
        print('#' + str(n) + ' pdf tree generated')


# ======================== GENERATE MODEL =========================

# ~~~~~~~~~~~~~~~~~~ Generate Model ~~~~~~~~~~~~~~~~~~~
def generate_model(model_dictionary, data_base, output_path, save_db=False, save_model=False, random_state=20):
    # If path is provided open database using pandas
    if '.csv' in str(data_base):
        print('\n~~~~~~~ SEARCHING DATABASE ~~~~~~~')
        print("Searching Database in", data_base)
        df_full = pd.read_csv(data_base, delimiter=';')
        print("Database loaded with", df_full.shape[1], 'parameters and', df_full.shape[0], 'lines')
    # If pandas dataframe is directly provided, skip the database reading
    elif 'pandas' in str(type(data_base)):
        print('\n~~~~~~~ DATABASE PROVIDED ~~~~~~~')
        df_full = data_base
        print("Database loaded with", df_full.shape[1], 'parameters and', df_full.shape[0], 'lines')
    else:
        print('\n~~~~~~~ INCORRECT DATABASE ~~~~~~~')
        print('No valid pandas dataframe or path was provided')
        return None, None, None

    # # -------------------INPUT -------------------
    print('\n~~~~~~~ INPUT PARAMETERS ~~~~~~~')
    for key, val in model_dictionary.items():
        print(key, '->', val)

    # # -------------------FILTER DATA TO TRAIN MODELS -------------------
    print('\n~~~~~~~ FILTERING DATABASE ~~~~~~~')
    df_model = df_full
    model_text = model_dictionary['model'] + '_'
    for key, val in model_dictionary.items():
        if key == 'model':
            pass
        elif key == 'label':
            model_text = model_text + 'L' + val.split('_')[-1] + '_'
            df_model['Y'] = df_model[val]
        elif key == 'class':
            model_text = model_text + 'C' + val
            if val == 'complete':
                df_model = filter_numerical_row_data(df_model, column_name='Y', upper_limit=11, remove=False)
            elif val == 'simple':
                df_model = filter_numerical_row_data(df_model, column_name='Y', upper_limit=9, remove=False)
            else:
                df_model = filter_numerical_row_data(df_model, column_name='Y', upper_limit=6, remove=False)
        elif key == 'balance':
            model_text = model_text + '_B' + val
        elif val != '':
            model_text = model_text + val + '_'
            keep_value = val[1:]
            # First char indicates the filter mode
            mode = val[0] == '-'
            df_model = filter_categorical_row_data(df_model, column_name=key, value=keep_value, remove=mode)
        else:
            model_text = model_text + 'all' + '_'

    print('Database generated:', model_text)
    if save_db:
        model_out_path = output_path + model_text + '.csv'
        df_model.to_csv(model_out_path, sep=';', mode='w', header=True, index=False)
        print('New database file generated:', model_out_path)

    # # -------------------PREPARE DATA TO TRAIN AND TEST THE MODELS -------------------
    print('\n~~~~~~~ GENERATING TRAINING DATA ~~~~~~~')
    remove_list = ['label', 'inf', 'ID', 'test_filename', 'original_filename', 'SP']
    X, _, y, _ = prepare_data(df_model, y_name='Y', test_size=0,
                              random_state=random_state,
                              keep_only_list=[],
                              remove_list=remove_list)

    # # -------------------SELECT THE MODEL -------------------
    print('\n~~~~~~~ MODEL SELECTION ~~~~~~~')
    model_name = model_dictionary['model'].split('_')[0]
    model_parameter = model_dictionary['model'].split('_')[1]

    if model_name == 'RFC':
        _, n, d = re.split('d|n', model_parameter)
        model = RandomForestClassifier(n_estimators=int(n), random_state=random_state, max_depth=int(d))

    print(model)
    model = train_model(model, X, y, model_name=output_path + model_text, save_model=save_model)

    return model, X, y
