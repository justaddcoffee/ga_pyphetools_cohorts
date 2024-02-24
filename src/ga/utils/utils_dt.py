import pandas as pd
import numpy as np
#from pyspark.sql.functions import *
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import (precision_recall_curve, auc, make_scorer, roc_curve,
                             recall_score, precision_score, f1_score)
#import matplotlib.pyplot as plt

#from bayes_opt import BayesianOptimization


# For a callable to be a scorer, it needs to meet the protocol specified by the following two rules: It can be called
# with parameters (estimator, X, y), where estimator is the model that should be evaluated, X is validation data,
# and y is the ground truth target for X (in the supervised case) or None (in the unsupervised case). It returns a
# floating point number that quantifies the estimator prediction quality on X, with reference to y. Again,
# by convention higher numbers are better, so if your scorer returns loss, that value should be negated. Advanced: If
# it requires extra metadata to be passed to it, it should expose a get_metadata_routing method returning the
# requested metadata. The user should be able to set the requested metadata via a set_score_request method. Please
# see User Guide and Developer Guide for more details. def my_aucpr_scorer(estimator, X, y): y_scores =
# estimator.predict_proba(X)[:, 1] # Compute precision-recall curve precision, recall, _ = precision_recall_curve(y,
# y_scores) # Compute area under the curve (AUC) for precision-recall auc_score = auc(recall, precision) return
# auc_score


def my_aucpr_scorer(y, y_pred, **kwargs):
    #    y_pred = estimator.predict_proba(X)[:, 1]
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y, y_pred)
    # Compute area under the curve (AUC) for precision-recall
    auc_score = auc(recall, precision)
    return auc_score


def run_dt_algorithm(df_train: pd.DataFrame,
                     df_test: pd.DataFrame,
                     disease: str,
                     pt_id_col='person_id',
                     pt_label_col='patient_label',
                     tune_hyperparameters=False,
                     class_weight="balanced",
                     min_samples_split=2,
                     max_depth=2,
                     ndigits=4):
    print('DT training/testing for ', disease)

    # extract labels from the horizontal dataframe
    y_train = df_train[pt_label_col]
    y_test = df_test[pt_label_col]

    # remove person_id and label columns from the feature set
    x_train = df_train.drop(columns=[pt_id_col, pt_label_col])
    x_test = df_test.drop(columns=[pt_id_col, pt_label_col])

    print(sum(y_train), sum(y_test), sum(y_train == 0), sum(y_test == 0))

    if tune_hyperparameters:
        dt = DecisionTreeClassifier(class_weight=class_weight)

        param_grid = {
            'min_samples_split': 2,
            'max_depth': (5, 10, 50, 100, 'None')
        }

        # Create GridSearchCV
        grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, scoring='average_precision', cv=3)

        # Fit the grid search to the data
        grid_search.fit(x_train, y_train)

        # Print the best parameters and corresponding accuracy
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Accuracy: {:.2f}%".format(grid_search.best_score_ * 100))
        best_dt_model = grid_search.best_estimator_

    else:
        dt = DecisionTreeClassifier(min_samples_split=min_samples_split, class_weight=class_weight, max_depth=max_depth)
        best_dt_model = dt.fit(x_train, y_train)
        print(best_dt_model)

    # Evaluate the model on the test set
    y_scores = best_dt_model.predict_proba(x_test)[:, 1]

    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)

    # Compute area under the curve (AUC) for precision-recall
    aucpr_score = auc(recall, precision)

    fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
    auroc_score = auc(fpr, tpr)

    recall = recall_score(y_test, y_scores>0.5, average=None, zero_division=np.nan)[1]
    precision = precision_score(y_test, y_scores > 0.5, average=None, zero_division=np.nan)[1]
    f1 = f1_score(y_test, y_scores > 0.5, average=None, zero_division=np.nan)[1]

    # Plot the precision-recall curve
    #plt.figure(figsize=(8, 6))
    #plt.plot(recall, precision, label='Precision-Recall Curve (AUC = {:.2f})'.format(auc_score))
    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    #plt.title('Precision-Recall Curve')
    #plt.legend(loc='best')
    #plt.show()

    #plt.figure(figsize=(8, 6))
    #plt.boxplot([y_scores[y_test == 0], y_scores[y_test == 1]], notch=True)
    #plt.xlabel('classes')
    #plt.ylabel('Scores')
    #plt.title('Boxplots')
    #plt.legend(loc='best')
    #plt.show()
    print('DT')
    print('disease\ttrain-pos\ttrain-neg\ttest-pos\ttest-neg\tAUCPR\tAUROC\tsensitivity\tprecision\tf1')
    print(
        f'DISOK:\tDT\t{disease}\t{sum(y_train)}\t{sum(y_train == 0)}\t{sum(y_test)}\t{sum(y_test == 0)}\t{round(aucpr_score, ndigits=ndigits)}\t{round(auroc_score, ndigits=ndigits)}\t{round(recall, ndigits=ndigits)}\t{round(precision, ndigits=ndigits)}\t{round(f1, ndigits=ndigits)}')

    return sum(y_train), sum(y_train==0), sum(y_test), sum(y_test==0), aucpr_score, auroc_score, recall, precision, f1
