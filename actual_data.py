from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, accuracy_score
from imblearn.over_sampling import ADASYN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import hyperopt as hyp
from icecream import ic


def make_roc_curve(true_y, y_prob):
    # I love stealing code from random websites
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


def getting_xgb_space(space):
    reg = xgb.XGBRegressor(eta=space['eta'],
                           gamma=space['gamma'],
                           max_depth=int(space['max_depth']),
                           min_child_weight=space['min_child_weight'],
                           colsample_bytree=space['colsample_bytree'],
                           reg_lambda=space['reg_lambda'],
                           verbosity=1,
                           eval_metric=space['eval'],
                           early_stopping_rounds=500)
    x_train, x_test = space['x_train'], space['x_test']
    y_train, y_test = space['y_train'], space['y_test']
    # In space,,,,,, no one can hear you space
    evaluation = [(x_train, y_train), (x_test, y_test)]
    reg.fit(x_train, y_train, eval_set=evaluation, verbose=False)
    pred = reg.predict(x_test)
    auc = roc_auc_score(y_test, pred)
    print("Accuracy score: {}".format(auc))
    return {'loss': -auc, 'status': hyp.STATUS_OK}


# Reading in and cleaning data
stroke = pd.read_csv("C:/Users/Sean Hayes/Downloads/healthcare-dataset-stroke-data.csv")
stroke_cols = list(stroke)
stroke_cols.remove("id")
stroke_cols.remove("work_type")
stroke = stroke[stroke_cols]
stroke.dropna(inplace=True)

# Replacing the string values with numbers
married_dict = {"Yes": 1, "No": 0}
residence_dict = {"Rural": 1, "Urban": 0}
smoking_dict = {"smokes": 2, "formerly smoked": 1, "never smoked": 0, "Unknown": np.nan}
gender_dict = {"Male": 1, "Other": np.nan, "Female": -1}
all_dicts = [married_dict, residence_dict, smoking_dict, gender_dict]
for i in all_dicts:
    stroke.replace(i, inplace=True)
stroke.dropna(inplace=True)

# Separating results from the rest
stroke_cols.remove("stroke")
result_stroke = stroke[["stroke"]]
no_result_stroke = stroke[stroke_cols]

# Train/test split
nstroke_train, nstroke_test, stroke_train, stroke_test = model_selection.train_test_split(no_result_stroke, result_stroke, test_size=0.2)
stroke_train, stroke_test = np.ravel(stroke_train), np.ravel(stroke_test)

four_ratio_ada = ADASYN(sampling_strategy=0.5)
ada_x_stroke_train, ada_x_stroke_test, ada_y_stroke_train, ada_y_stroke_test = model_selection.train_test_split(no_result_stroke, result_stroke, test_size=0.2)
ada_x_stroke_train, ada_y_stroke_train = four_ratio_ada.fit_resample(ada_x_stroke_train, ada_y_stroke_train)
ada_y_stroke_train, ada_y_stroke_test = np.ravel(ada_y_stroke_train), np.ravel(ada_y_stroke_test)

reg_rf = RandomForestRegressor()
reg_rf.fit(ada_x_stroke_train, ada_y_stroke_train)
reg_rf_pred = reg_rf.predict(ada_x_stroke_test)
make_roc_curve(ada_y_stroke_test, reg_rf_pred)
print(f'model 1 AUC score: {roc_auc_score(ada_y_stroke_test, reg_rf_pred)}')
# lol rf has better auc than boost model. Guess I didn't tune the hyperparameters enough

# Now for the main event, my stumbling my way through using xgb
trials = hyp.Trials()
xgb_space = {'eta': hyp.hp.loguniform('eta', -20, 0),
             'gamma': hyp.hp.uniform('gamma', 0.1, 40),
             'max_depth': hyp.hp.quniform('max_depth', 2, 10, 1),
             'min_child_weight': hyp.hp.uniform('min_child_weight', 0.5, 10),
             'colsample_bytree': hyp.hp.uniform('colsample_bytree', 0.1, 1),
             'reg_lambda': hyp.hp.uniform('reg_lambda', 0.5, 5),
             'x_train': ada_x_stroke_train,
             'x_test': ada_x_stroke_test,
             'y_train': ada_y_stroke_train,
             'y_test': ada_y_stroke_test,
             'eval': 'auc'}
best_xgb_hyperparameters = hyp.fmin(fn=getting_xgb_space,
                                    space=xgb_space,
                                    algo=hyp.tpe.suggest,
                                    max_evals=200,
                                    trials=trials)
print(best_xgb_hyperparameters)
best_xgb_hyperparameters['max_depth'] = int(best_xgb_hyperparameters['max_depth'])
best_xgb = xgb.XGBRegressor(**best_xgb_hyperparameters)
best_xgb.fit(ada_x_stroke_train, ada_y_stroke_train)
best_xgb_pred = best_xgb.predict(ada_x_stroke_test)
make_roc_curve(ada_y_stroke_test, best_xgb_pred)
print(f'XGBoost Model AUC score: {roc_auc_score(ada_y_stroke_test, best_xgb_pred)}')

