from utils import *

import datetime
import pandas as pd
import pickle

from catboost.utils import get_gpu_device_count
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from xgboost import XGBClassifier


# 載入資料
df_a, df_b, df_c, df_d, df_e, df_f = load_data('data')

# 合併所有資料
df_list = [df_a, df_b, df_c, df_d, df_e, df_f]

# 訓練日期
today = datetime.date.today().strftime( '%Y%m%d' )

# 作業根目錄
root_dir = "."

# 取得檔名
def getFilename(filetype: str, filename: str, fileext: str):
    """
    取得工作檔路徑
    """
    workdir = os.path.join(root_dir, filetype)
    try:
        os.makedirs(workdir)
    except:
        pass
    
    return os.path.join( workdir, f'{filename}-{today}.{fileext}' )


# 測試資料比例
# test_size_list = [ .25 ]
test_size_list = [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .80, .85, .90, .95]

# 隨機種子
random_seed_list = [3, 106, 2019]
# random_seed_list = [3]

# 檢查是否有 GPU 支援
task_type = "GPU" if get_gpu_device_count() > 0 else "CPU"


model_list = [
    BernoulliNB(),
    CatBoostClassifier(task_type = task_type),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    ExtraTreesClassifier(n_jobs=-1),
    GaussianNB(),
    GradientBoostingClassifier(),
    GaussianProcessClassifier(multi_class = 'one_vs_rest', n_jobs = -1), # Out of memory
    KNeighborsClassifier(n_jobs = -1), 
    LGBMClassifier(n_jobs = -1), 
    # LabelSpreading(n_jobs = -1),          # Out of Memory
    LinearDiscriminantAnalysis(),
    LogisticRegression(multi_class='multinomial', n_jobs=-1) ,
    LogisticRegressionCV(multi_class='multinomial', n_jobs=-1) ,
    MLPClassifier(hidden_layer_sizes=(96, 48, 24, 12,), max_iter=450),
    NearestCentroid(),
    # LabelPropagation(n_jobs = -1),        # Out of Memory
    PassiveAggressiveClassifier(n_jobs = -1),
    Perceptron(n_jobs = -1),
    QuadraticDiscriminantAnalysis(),
    # RadiusNeighborsClassifier(n_jobs = -1),
    RandomForestClassifier(n_jobs = -1),
    RidgeClassifier(),
    RidgeClassifierCV(),
    XGBClassifier(n_jobs = -1),
]


# 訓練報告總記錄
train_reports = []

for df in df_list:

    scenario = df.iloc[0,-1]

    for test_size in test_size_list:
        for random_seed in random_seed_list:
            
            # Cartesian - 笛卡兒坐標系（英語：Cartesian coordinate system)
            coordinate = 'cartesian'
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:106], df.iloc[:,-2], test_size = test_size, random_state = random_seed)

            for base_model in model_list:

                model = clone(base_model)
                model.fit(X_train, y_train)

                model_name = model.__class__.__name__
                
                train_metrics = {
                    "model_name": model_name,
                    "random_seed": random_seed,
                    "test_size": test_size,
                    "train_count": len(y_train),
                    "test_count": len(y_test),
                    'dataset_shape': X_train.shape,
                    "scenario": scenario,
                    "type": "experiment",
                    "data_type": ['real', 'imaginary'],
                    "coordinate": coordinate,
                }
                
                y_predict = model.predict(X_train)
                # y_predict_proba = model.predict_proba(X_train)
                report = classification_report(y_train, y_predict, output_dict = True)
                kappa = cohen_kappa_score(y_train, y_predict)
                
                cm = confusion_matrix(y_train, y_predict)
                labels = list(set(y_train))
                labels.sort()
                    
                train_metrics['train_report'] = report
                train_metrics['train_kappa'] = kappa
                train_metrics['train_cm'] = cm
                train_metrics['train_category'] = labels
                    
                y_predict = model.predict(X_test)
                # y_predit_proba = model.predict_proba(X_test)
                report = classification_report(y_test, y_predict, output_dict = True)
                kappa = cohen_kappa_score(y_test, y_predict)
                    
                cm = confusion_matrix(y_test, y_predict)
                labels = list(set(y_test))
                labels.sort()
                    
                train_metrics['test_report'] = report
                train_metrics['test_kappa'] = kappa
                train_metrics['test_cm'] = cm
                train_metrics['test_category'] = labels
                    
                train_reports.append(train_metrics)

            # Polar - 極坐標系（英語：Polar coordinate system）
            coordinate = 'polar'
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,106:212], df.iloc[:,-2], test_size = test_size, random_state = random_seed)

            for base_model in model_list:

                model = clone(base_model)
                model.fit(X_train, y_train)

                model_name = model.__class__.__name__
                
                train_metrics = {
                    "model_name": model_name,
                    "random_seed": random_seed,
                    "test_size": test_size,
                    "train_count": len(y_train),
                    "test_count": len(y_test),
                    'dataset_shape': X_train.shape,
                    "scenario": scenario,
                    "type": "experiment",
                    "data_type": ['amplitude', 'phase'],
                    "coordinate": coordinate,
                }
                
                y_predict = model.predict(X_train)
                # y_predict_proba = model.predict_proba(X_train)
                report = classification_report(y_train, y_predict, output_dict = True)
                kappa = cohen_kappa_score(y_train, y_predict)
                
                cm = confusion_matrix(y_train, y_predict)
                labels = list(set(y_train))
                labels.sort()
                    
                train_metrics['train_report'] = report
                train_metrics['train_kappa'] = kappa
                train_metrics['train_cm'] = cm
                train_metrics['train_category'] = labels
                    
                y_predict = model.predict(X_test)
                # y_predit_proba = model.predict_proba(X_test)
                report = classification_report(y_test, y_predict, output_dict = True)
                kappa = cohen_kappa_score(y_test, y_predict)
                    
                cm = confusion_matrix(y_test, y_predict)
                labels = list(set(y_test))
                labels.sort()
                    
                train_metrics['test_report'] = report
                train_metrics['test_kappa'] = kappa
                train_metrics['test_cm'] = cm
                train_metrics['test_category'] = labels
                    
                train_reports.append(train_metrics)

            # 全部資料
            coordinate = 'full'
            X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:-2], df.iloc[:,-2], test_size = test_size, random_state = random_seed)

            for base_model in model_list:

                model = clone(base_model)
                model.fit(X_train, y_train)

                model_name = model.__class__.__name__
                
                train_metrics = {
                    "model_name": model_name,
                    "random_seed": random_seed,
                    "test_size": test_size,
                    "train_count": len(y_train),
                    "test_count": len(y_test),
                    'dataset_shape': X_train.shape,
                    "scenario": scenario,
                    "type": "experiment",
                    "data_type": ['real', 'imaginary', 'amplitude', 'phase'],
                    "coordinate": coordinate,
                }
                
                y_predict = model.predict(X_train)
                # y_predict_proba = model.predict_proba(X_train)
                report = classification_report(y_train, y_predict, output_dict = True)
                kappa = cohen_kappa_score(y_train, y_predict)
                
                cm = confusion_matrix(y_train, y_predict)
                labels = list(set(y_train))
                labels.sort()
                    
                train_metrics['train_report'] = report
                train_metrics['train_kappa'] = kappa
                train_metrics['train_cm'] = cm
                train_metrics['train_category'] = labels
                    
                y_predict = model.predict(X_test)
                # y_predit_proba = model.predict_proba(X_test)
                report = classification_report(y_test, y_predict, output_dict = True)
                kappa = cohen_kappa_score(y_test, y_predict)
                
                cm = confusion_matrix(y_test, y_predict)
                labels = list(set(y_test))
                labels.sort()
                    
                train_metrics['test_report'] = report
                train_metrics['test_kappa'] = kappa
                train_metrics['test_cm'] = cm
                train_metrics['test_category'] = labels
                    
                train_reports.append(train_metrics)

pickle_name = getFilename("pickle", f"exp-all", "pickle")
pickle.dump( train_reports, open( pickle_name, "wb" ) )
