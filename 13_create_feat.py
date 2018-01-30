import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

data_path = '../data/'

train = pd.read_csv(data_path+'d_train_20180102.csv',encoding='gb2312')
test = pd.read_csv(data_path+'d_test_A_20180102.csv',encoding='gb2312')

def min_max_normalize(df, name):
    # 归一化
    max_number = df[name].max()
    min_number = df[name].min()
    # assert max_number != min_number, 'max == min in COLUMN {0}'.format(name)
    df[name] = df[name].map(lambda x: float(x - min_number + 1) / float(max_number - min_number + 1))
    # 做简单的平滑,试试效果如何
    return df


def cheng_feat(train,a,b):
    train[a+"*"+b] = pd.DataFrame((train[a]*train[b]))
    return train

def chu_feat(train,a,b):
    train[a+"/"+b] = pd.DataFrame((train[a]/train[b]))
    return train

def jia_feat(train,a,b):
    train[a+"+"+b] = pd.DataFrame((train[a]+train[b]))
    return train

def jian_feat(train,a,b):
    train[a+"-"+b] = pd.DataFrame((train[a]-train[b]))
    return train

def log_feat(train,a):
    train["log:"+a] = train[a].apply(lambda x: np.log(x))
    return train

def size_feat(train,a,b):
    train = cheng_feat(train,a,b)
    train = chu_feat(train,a,b)
    train = jia_feat(train,a,b)
    train = jian_feat(train,a,b)
    train = log_feat(train,a)
    return train

def zaotezheng(data):
    predictors2 = [f for f in data.columns if f not in ['id','血糖',"性别"]]
    predictors3 = [f for f in data.columns if f not in ['id','血糖',"性别"]]
    for i in predictors2:
        for j in predictors3:
            data = size_feat(data,i,j)
    
    predictors1 = [f for f in data.columns if f not in ['血糖','id']]
    for i in predictors1:
        min_max_normalize(data,i)
    return data
  
def make_feat(train,test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train,test])

    data['性别'] = data['性别'].map({'男':1,'女':0})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days
    
    data.fillna(data.median(axis=0),inplace=True)
#    data = data[data["血糖"]<35]
#加入我自己的处理：去除乙肝指标，然后试着多项式特征
    data.drop(['体检日期','乙肝表面抗原','乙肝表面抗体','乙肝e抗原','乙肝e抗体','乙肝核心抗体'],axis=1,inplace=True)
#先标准化数据，然后移除方差小于thresh的特征：
#    StandardScaler().fit_transform(data)
    data = data[data["血糖"]<35]
    data = zaotezheng(data)
    
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    return train_feat,test_feat


        

train_feat,test_feat = make_feat(train,test)

predictors = [f for f in test_feat.columns if f not in ['id','血糖']]



def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label,pred)*0.5
    return ('mse',score,False)

print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.7,
    'num_leaves': 60,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

print('开始CV 5折训练...')
scores = []
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))
kf = KFold(len(train_feat), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['血糖'],categorical_feature=['性别'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['血糖'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=100)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += gbm.predict(train_feat2[predictors])
    test_preds[:,i] = gbm.predict(test_feat[predictors])
print('线下得分：    {}'.format(mean_squared_error(train_feat['血糖'],train_preds)*0.5))
print('CV训练用时{}秒'.format(time.time() - t0))
print(train_feat.info())
#submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
#submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),header=None,
#                  index=False, float_format='%.4f')



predictors1 = feat_imp.index[:60]
train_feat3 = pd.DataFrame(train_feat[predictors1])
train_feat3["血糖"] = pd.DataFrame(train_feat["血糖"])
train_feat3["性别"] = pd.DataFrame(train_feat["性别"])
train_feat3["id"] = pd.DataFrame(train_feat["id"])
test_feat3 = pd.DataFrame(test_feat[predictors1])
test_feat3["血糖"] = pd.DataFrame(test_feat["血糖"])
test_feat3["性别"] = pd.DataFrame(test_feat["性别"])
test_feat3["id"] = pd.DataFrame(test_feat["id"])

train_feat3.to_csv("train_feat3.csv")
test_feat3.to_csv("test_feat3.csv")

