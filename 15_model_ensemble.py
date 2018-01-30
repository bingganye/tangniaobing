import time
start_time = time.time()
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn import grid_search
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor


def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5

RMSE = make_scorer(mean_squared_error_, greater_is_better=False)


def xx(x):
    max_x = max(x) * 0.95
    qut_x =  max(x) * 0.9
    min_x = max(x) * 0.25

    for xi in x:
        if xi >= max_x:
            xi = xi * 1.5
        elif (xi < max_x) & (xi >= qut_x):
            xi = xi * 1.25
        elif xi <= min_x:
            xi = xi * 0.75
        else:
            xi = xi
    return x



class Ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models    
    
    def fit_predict(self,X,y,T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        
        folds = list(KFold(len(y),n_folds = self.n_folds, shuffle=True, random_state=520))
        
        S_train = np.zeros((X.shape[0],len(self.base_models)))
        S_test = np.zeros((T.shape[0],len(self.base_models)))
        
        for i, clf in enumerate(self.base_models):
            print('Fitting For Base Model #{0} / {1} ---'.format(i+1, len(self.base_models)))

            S_test_i = np.zeros((T.shape[0],len(folds)))
            
            for j, (train_idx, test_idx) in enumerate(folds):
                print('--- Fitting For Fold #{0} / {1} ---'.format(j+1, self.n_folds))

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T)[:]
                
                print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

            S_test[:, i] = S_test_i.mean(1)

            print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        print('--- Base Models Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))


        param_grid = {
            'n_estimators': [100],
            'learning_rate': [0.05],
            'subsample': [0.75]
        }
        grid = grid_search.GridSearchCV(estimator=self.stacker, param_grid=param_grid, n_jobs=1, cv=5, verbose=20, scoring=RMSE)
        grid.fit(S_train, y)

        # a little memo
        message = 'to determine local CV score of #28'

        try:
            print('Param grid:')
            print(param_grid)
            print('Best Params:')
            print(grid.best_params_)
            print('Best CV Score:')
            print(-grid.best_score_)
            print('Best estimator:')
            print(grid.best_estimator_)
            print(message)
        except:
            pass

        print('--- Stacker Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        y_pred = grid.predict(S_test)[:]

        return y_pred


def main():
    train = pd.read_csv('train_feat3.csv',encoding='gb2312')
    test = pd.read_csv('test_feat3.csv',encoding='gb2312')

    
    X_train = train[:]
    y_train = train['血糖'].values
    X_test = test[:]
    cols_to_drop = ['id', '血糖']
    for col in cols_to_drop:
        try:
            X_train.drop(col, axis=1, inplace=True)
            X_test.drop(col, axis=1, inplace=True)
        except:
            continue

    print('--- Features Set: %s minutes ---' % round(((time.time() - start_time) / 60), 2))
    print('Number of Features: ', len(X_train.columns.tolist()))

    base_models = [
        RandomForestRegressor(
            n_jobs=1, random_state=2016, verbose=1,
            n_estimators=500, max_features=12
        ),
        ExtraTreesRegressor(
            n_jobs=1, random_state=2016, verbose=1,
            n_estimators=500, max_features=12
        ),
        GradientBoostingRegressor(
            random_state=2016, verbose=1,
            n_estimators=500, max_features=12, max_depth=8,
            learning_rate=0.05, subsample=0.8
        ),
        XGBRegressor(
            seed=2016,
            n_estimators=200, max_depth=8,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.85
        )
    ]
    ensemble = Ensemble(
        n_folds=5,
        stacker=GradientBoostingRegressor(
            random_state=2016, verbose=1
        ),
        base_models=base_models
    )

    y_pred = ensemble.fit_predict(X=X_train, y=y_train, T=X_test)

    pd.DataFrame({'血糖': y_pred}).to_csv('submission_ensemble(notxx).csv', index=False,header=False )

    print('--- Submission Generated: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

if __name__ == '__main__':
    main()


