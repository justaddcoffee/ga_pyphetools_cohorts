import pandas as pd
#from pyspark.sql.functions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve, auc, make_scorer
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization


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


def my_aucpr_scorer(y, y_pred, needs_proba=True, **kwargs):
    #    y_pred = estimator.predict_proba(X)[:, 1]
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y, y_pred)
    # Compute area under the curve (AUC) for precision-recall
    auc_score = auc(recall, precision)
    return auc_score


def vertical_to_horizontal(df):
    print(df.columns)

    df = df.drop_duplicates()

    # Group by "person_id" and "patient_label" and pivot on "hpo_term_id"
    df_pivoted = df.drop(columns=['hpo_term_label', 'negated', 'weight']).groupby(
        ["person_id", "patient_label", "hpo_term_id"]).size().unstack(fill_value=0).reset_index()

    print(df_pivoted.columns)

    return df_pivoted


def run_rf_algorithm(pt_train_df: pd.DataFrame,
                     pt_test_df: pd.DataFrame,
                     disease: str,
                     tune_hyperparameters=False,
                     n_estimators=500,
                     min_samples_split=2):
    print('training/testing for ', disease)

    ids_train = pt_train_df.drop(columns=['hpo_term_label', 'negated', 'weight', 'hpo_term_id']).drop_duplicates()
    ids_test = pt_test_df.drop(columns=['hpo_term_label', 'negated', 'weight', 'hpo_term_id']).drop_duplicates()
    pt_df_h = vertical_to_horizontal(pd.concat([pt_train_df, pt_test_df]))

    x_train = pt_df_h.merge(ids_train, on=['person_id', 'patient_label'], how='inner')
    x_test = pt_df_h.merge(ids_test, on=['person_id', 'patient_label'], how='inner')

    y_train = x_train['patient_label']
    y_test = x_test['patient_label']

    print("TRAIN: Number of positives = ", np.sum(y_train), " negatives = ", np.sum(y_train == 0))
    print("TEST: Number of positives = ", np.sum(y_test), " negatives = ", np.sum(y_test == 0))

    x_train = x_train.drop(columns=['person_id', 'patient_label'])
    x_test = x_test.drop(columns=['person_id', 'patient_label'])

    if tune_hyperparameters:
        rf = RandomForestClassifier()

        param_grid = {
            'n_estimators': [100, 500, 1000],  # You can add more values to this list
            'min_samples_split': 2
        }

        # Create GridSearchCV
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='average_precision', cv=3)

        # Fit the grid search to the data
        grid_search.fit(x_train, y_train)

        # Print the best parameters and corresponding accuracy
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Accuracy: {:.2f}%".format(grid_search.best_score_ * 100))
        best_rf_model = grid_search.best_estimator_

    else:
        rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split)
        best_rf_model = rf.fit(x_train, y_train)

    # Evaluate the model on the test set
    y_scores = best_rf_model.predict_proba(x_test)[:, 1]

    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)

    # Compute area under the curve (AUC) for precision-recall
    auc_score = auc(recall, precision)

    # Plot the precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve (AUC = {:.2f})'.format(auc_score))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.boxplot([y_scores[y_test == 0], y_scores[y_test == 1]], notch=True)
    plt.xlabel('classes')
    plt.ylabel('Scores')
    plt.title('Boxplots')
    plt.legend(loc='best')
    plt.show()

    return auc_score


def run_rf_algorithm_cv(pt_df,
                        n_cv_folds=5,
                        niters=5,
                        n_estimators=100,
                        criterion='gini',
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        min_weight_fraction_leaf=0.0,
                        max_features='sqrt',
                        max_leaf_nodes=None,
                        min_impurity_decrease=0.0,
                        bootstrap=True,
                        oob_score=False,
                        n_jobs=None,
                        random_state=41,
                        verbose=0,
                        warm_start=False,
                        class_weight=None,
                        ccp_alpha=0.0,
                        max_samples=None):
    ids_train = pt_df.drop(columns=['hpo_term_label', 'negated', 'weight', 'hpo_term_id']).drop_duplicates()
    pt_df_h = vertical_to_horizontal(pt_df)
    x_train = pt_df_h.merge(ids_train, on=['person_id', 'patient_label'], how='inner')
    y_train = x_train['patient_label']

    print("TRAIN: Number of positives = ", np.sum(y_train), " negatives = ", np.sum(y_train == 0))

    x_train = x_train.drop(columns=['person_id', 'patient_label'])

    rf = RandomForestClassifier(n_estimators=n_estimators,
                                criterion=criterion,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_features=max_features,
                                max_leaf_nodes=max_leaf_nodes,
                                min_impurity_decrease=min_impurity_decrease,
                                bootstrap=bootstrap,
                                oob_score=oob_score,
                                n_jobs=n_jobs,
                                random_state=random_state,
                                verbose=verbose,
                                warm_start=warm_start,
                                class_weight=class_weight,
                                ccp_alpha=ccp_alpha,
                                max_samples=max_samples)

    aucpr_scorer = make_scorer(my_aucpr_scorer)
    aucpr_val = 0
    for ncviter in range(niters):
        aucpr_val = aucpr_val + np.mean(cross_val_score(rf, X=x_train, y=y_train, cv=n_cv_folds, scoring=aucpr_scorer))
        print(".")

    return aucpr_val / niters


def optimize_rfc(pt_df,
                 pbounds,  # parameter space for bayesian optimization
                 n_cv_folds=3,  # number of stratified folds for internal cv evaluation
                 niters=1,  # number of itaratons of stratified folds for internal cv evaluation
                 nBayes_iters=60,  #number of Bayesian optimization iterations
                 init_points=21,  #number of points to evaluate to init the estimate of the objective function used by Bayes opt
                 ):
    """Apply Bayesian Optimization to Random Forest parameters."""

    def rfc_crossval(n_estimators, min_samples_split, pt_df=pt_df, n_cv_folds=n_cv_folds, niters=niters):
        """Wrapper of RandomForest cross validation.

        Notice how we ensure n_estimators and min_samples_split are cast
        to integer before we pass them along. Moreover, to avoid max_features
        taking values outside the (0, 1) range, we also ensure it is capped
        accordingly.
        """
        return run_rf_algorithm_cv(pt_df=pt_df,
                                   n_cv_folds=n_cv_folds,
                                   niters=niters,
                                   n_estimators=int(n_estimators),
                                   min_samples_split=int(min_samples_split))

    optimizer = BayesianOptimization(
        f=rfc_crossval,
        pbounds=pbounds,
        random_state=1234,
        verbose=2
    )
    print(optimizer)
    optimizer.set_gp_params(alpha=1e-2)
    optimizer.maximize(n_iter=nBayes_iters, init_points=init_points)

    print("Final result:", optimizer.max)
    return optimizer.max
