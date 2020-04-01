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

from sklearn.neural_network import MLPClassifier

# 載入資料
df_a, df_b, df_c, df_d, df_e, df_f = load_data('data')

# 合併所有資料
df = pd.concat([df_a, df_b, df_c, df_d, df_e, df_f], ignore_index = True)

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
test_size = .25

# 隨機種子
random_seed = 3

# 重覆訓練次數
loops = 10

# 待驗證模型
base_model = MLPClassifier()
model_name = base_model.__class__.__name__

# 訓練報告總記錄
train_reports = []

# 訓練資料 / 驗證資料比例： 3:1
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:106], df.iloc[:,-2], test_size = test_size, random_state = random_seed)

# 開始訓練並記錄每次訓練的權重及成效
for n in range(loops):

    model = clone(base_model)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    report = classification_report(y_test, y_predict, output_dict = True)
    kappa = cohen_kappa_score(y_test, y_predict)
    
    cm = confusion_matrix(y_test, y_predict)

    train_metrics = {
        "model_name": model_name,
        "random_seed": random_seed,
        "test_size": test_size,
        "train_count": len(y_train),
        "test_count": len(y_test),
        'dataset_shape': X_train.shape,
        'test_index': n,
        'report': report,
        'kappa': kappa,
        'cm': cm,
        'coefs': model.coefs_,
        'loss': model.loss_,
        'n_iter': model.n_iter_,
    }

    train_reports.append(train_metrics)


pickle_name = getFilename("pickle", f"mlp-weight", "pickle")
pickle.dump( train_reports, open( pickle_name, "wb" ) )
