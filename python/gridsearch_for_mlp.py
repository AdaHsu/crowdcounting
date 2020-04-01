from utils import *

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import datetime
import os 


# 載入資料
df_a, df_b, df_c, df_d, df_e, df_f = load_data('data')


# 練習
# df = df_a

# 合併所有資料
df = pd.concat([df_a, df_b, df_c, df_d, df_e, df_f], ignore_index = True)

today = datetime.date.today().strftime( '%Y%m%d' )

root_dir = "."

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
test_size_list = [ .25 ]
# test_size_list = [.25, .35, .45, .55, .65, .75, .80, .85, .90, .95]

# 隨機種子
# random_seed_list = [3, 106, 2019]
random_seed_list = [3]

hidden_layer_sizes = [(100,), (96,48,24,12,), (48,24,12,), (24,12,)]
max_iters = [100,150,200,250,300,350,400,450,500]
# max_iters = [100]

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:106], 
                                                    df.iloc[:,-2], 
                                                    test_size = test_size_list[0], 
                                                    random_state = random_seed_list[0])

mlp = MLPClassifier()
parameters = {'hidden_layer_sizes': hidden_layer_sizes, 'max_iter': max_iters}
clf = GridSearchCV(estimator = mlp, 
                   param_grid = parameters, 
                   scoring = make_scorer(cohen_kappa_score, greater_is_better=True),
                   n_jobs = -1)
clf.fit(X_train, y_train)

pickle_name = getFilename("pickle", "mlp_gridsearch", "pickle")
pickle.dump( clf.cv_results_, open( pickle_name, "wb" ) )