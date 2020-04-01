"""
    碩專論文 Python 工具組
"""
# Line SDK
# from linebot import LineBotApi
# from linebot.models import TextSendMessage
# Excel SDK
# from openpyxl import load_workbook
# from openpyxl import Workbook
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import re
import scipy.io
import seaborn as sns
import sys

def filter_outlier(df: pd.DataFrame, outlier: pd.DataFrame, columns: List[int]):
    """
    檢查 outlier 指定的 columns 中沒有 outlier 的索引值，然後只傳回 df 內這些索引值的資料

    :param df: 原始資料
    :param outlier: 原始資料中各欄位是否為 outlier 的對照表
    :param columns: df 中要檢查的欄位項

    :returns:
        - not_outlier: pd.DataFrame 不是 離群值 的資料集合
        - outlier: pd.DataFrame 是 離群值 的資料集合
    """

    # 先取得哪些資料的欄位都不是 outlier 的資料索引
    not_outlier = [idx for idx in outlier.index if not any(outlier.iloc[idx, columns])]
    outlier = [idx for idx in outlier.index if any(outlier.iloc[idx, columns])]
    # 把不是 outlier 的資料索引對應的資料傳回去
    return df.iloc[not_outlier], df.iloc[outlier]

def check_outlier(df: pd.DataFrame):
    """
    計算 df 中各筆資料、各欄位是否為高度干涉（以 上四分位數 + 1.5 倍 4 分位距為有效值）

    :param df: 待計算的資料集
    :return: outlier_df: 與原 df 對應的 outlier 檢查結果
    """

    outlier = pd.DataFrame()

    for idx in df.columns:
        if df[idx].dtype != np.object:
            # 下四分位數
            q1 = df[idx].quantile(0.25)
            # 上四分位數
            q3 = df[idx].quantile(0.75)

            # 四分位距
            iqr = q3 - q1
            max = q3 + 1.5*iqr
            # min = q1 - 1.5*iqr
            # outlier[idx] = [True if x > max or x < min else False for x in df[idx]]
            outlier[idx] = [True if x > max else False for x in df[idx]]
        
    return outlier            

def create_df(filename: str, scenario: str):
    """
    將指定 CSI 檔案讀入後，計算振幅及相位角並加入 DataFrame 內
    
    :param filename: 要載入的 CSI 檔案
    :param scenario: 該 CSI 檔案的情境標籤

    :return: dataframe: 計算後的 DataFrame
        - column[0:53]: 實部資料
        - column[53:106]: 虛部資料
        - column[106:159]: 振幅
        - column[159:212]: 相位
    """

    # 載入 CSI 檔
    mat = scipy.io.loadmat(filename).get("CSI")
    # 轉成 DataFrame
    tmp = pd.DataFrame(data = mat)
    df = tmp.iloc[:, :-1]
    df_abs = pd.DataFrame()
    df_phase = pd.DataFrame()

    for i in range(0, 53):
        polar = [(abs(complex(x,y)), np.angle(complex(x,y)) ) for x,y in zip(df[i], df[i+53])]
        df_abs[len(df.columns) + i] = [p[0] for p in polar]
        df_phase[len(df.columns) + i + 53] = [p[1] for p in polar] 

    df = pd.concat( [df, df_abs, df_phase], axis = 1)
    df[len(df.columns)] = tmp.iloc[:,-1].astype(int)
    df[len(df.columns)] = scenario

    return df


def load_data(path: str = "."):
    """
    :param path: 資料來源目錄

    :returns: 
        - df_a: pd.DataFrame, Scenario_A.mat 原始 CSI 欄位內容 + 53 欄振幅 + 53 欄相位 + 人數 + 英文字母場景標籤
        - df_b: pd.DataFrame, Scenario_B.mat 原始 CSI 欄位內容 + 53 欄振幅 + 53 欄相位 + 人數 + 英文字母場景標籤
        - df_c: pd.DataFrame, Scenario_C.mat 原始 CSI 欄位內容 + 53 欄振幅 + 53 欄相位 + 人數 + 英文字母場景標籤
        - df_d: pd.DataFrame, Scenario_D.mat 原始 CSI 欄位內容 + 53 欄振幅 + 53 欄相位 + 人數 + 英文字母場景標籤
        - df_e: pd.DataFrame, Scenario_E.mat 原始 CSI 欄位內容 + 53 欄振幅 + 53 欄相位 + 人數 + 英文字母場景標籤
        - df_f: pd.DataFrame, Scenario_F.mat 原始 CSI 欄位內容 + 53 欄振幅 + 53 欄相位 + 人數 + 英文字母場景標籤
    """

    # 載入原始資料

    df_a = create_df( os.path.join(path, 'Scenario_A.mat'), 'A' )
    df_b = create_df( os.path.join(path, 'Scenario_B.mat'), 'B' )
    df_c = create_df( os.path.join(path, 'Scenario_C.mat'), 'C' )
    df_d = create_df( os.path.join(path, 'Scenario_D.mat'), 'D' )
    df_e = create_df( os.path.join(path, 'Scenario_E.mat'), 'E' )
    df_f = create_df( os.path.join(path, 'Scenario_F.mat'), 'F' )

    # 回傳
    return df_a, df_b, df_c, df_d, df_e, df_f



def check_excel(filename: str, columns: List[str] = None):
    """
    檢查指定 Excel 是否存在，若不存在則建立之
    
    :param filename: Excel 檔檔名
    :param columns: 首列欄位名稱
    :return: workbook Excel 物件
    """
    if not os.path.exists(filename):
        wb = Workbook()
        ws = wb.active
        ws.title = 'Summary'
        if columns is not None:
            summary_header = columns
            ws.append(summary_header)
        wb.save(filename)
    else:
        wb = load_workbook(filename)
        
    return wb



# def get_work_sheet(workbook: Workbook, sheet_name: str, columns: List[str] = None, 
#                     formula: str = None, skip_column: int = 0, reserved_column: int = 0):
#     """
#     自 Excel Workbook 物件中取得指定名稱/Title的 work sheet，如果指定的 work sheet 不存在則建立之。建立時自動導入第一行公式
    
#     :param workbook: Excel Workbook
#     :param sheet_name: 指定的 sheet_name
#     :param columns: 欄位名稱，如果 formula 存在則放在第二列
#     :param formula: 預設放在首列的公式
#     :param skip_column: 前面幾個欄位跳過不用放公式
#     :param reserved_column: 後面幾個欄位跳過不用放公式
#     """
    
#     if sheet_name not in workbook.sheetnames:
#         workbook.create_sheet(sheet_name)
        
#         if columns is not None and formula is not None:
#             if skip_column > 0:
#                 formula_row = [None for i in range(skip_column)]
#             else:
#                 formula_row = []
            
#             for i in range(len(columns) - skip_column - reserved_column):
#                 formula_row.append( formula )

#             workbook[sheet_name].append(formula_row)

#         if columns is not None:    
#             workbook[sheet_name].append(columns)
        
#     return workbook[sheet_name]


def prepare_path():
    """
    依執行檔名稱建立對應的保存目錄及 EXCEL 檔名
    
    :returns:
        - dir_name: 作業保存目錄，絕對路徑
        - excel_name: 訓練記錄檔 EXCEL 檔名，絕路徑
    """

    dir_name = os.path.splitext(sys.argv[0])[0]
    try:
        os.makedirs(dir_name)
    except:
        pass

    excel_path = os.path.join(os.path.dirname(sys.argv[0]), 'excel')
    
    excel_name = os.path.join(excel_path, f"CSI-{dir_name}.xlsx")

    return os.path.abspath(dir_name), os.path.abspath(excel_name)


def create_path(path_name: str):
    """
    建立指定目錄

    :param path_name: 指定目錄
    """

    if len(path_name) > 0:
        try:
            os.makedirs(path_name)
        except:
            pass
        
# def line_message(message: str):
#     """
#     傳送 Line 訊息給作者本人

#     :param message: 文字訊息內容
#     """

#     if len(message) > 0:

#         print(message)
#         return
        
#         try:
#             line = LineBotApi('zYR3vcx4eAs246b7t7kjZtJ0XMYlzXzO9bzSmEXqcyr2+7o7Mme+AaEch9HvJPYBody/V+gZpW7MY050Wzw5nkhI7kTEZ55CIwCX0yzgoI+FnWWvYMlAGgdev9PQbxL5JV53MRPtgDzdzaJNyhJeoAdB04t89/1O/w1cDnyilFU=')
#             line.push_message('U10923dff73639851f791ab1987120883',
#                                 TextSendMessage(text=message))
#         except:
#             pass

def is_jupyter_notebook():
    """
    檢查是否位於 jupyter-notebook 環境下

    :return: is_jupyter_notebook: 是否處於 jupyter-notebook 環境
    """

    result = True if 'notebook' in os.environ['_'] else False

    return result


def grep_classification_report(reports: List, conditions: Dict, sort_by: Dict = None):
    """
    篩選報告
    
    :param reports: 模組運算成果
    :param conditions: 欲篩選的欄位條件，可多條件並存
    :param sort_by: 排序條件，只限一個條件 
    """
    
    result = []
    
    for report in reports:
        matched = True
        for key in conditions.keys():
            if report[key] != conditions[key]:
                matched = False
                break
        if matched:
            result.append(report)

    if sort_by is not None:
        key = list(sort_by.keys())[0]
        reverse = True if sort_by[key].lower() == 'descending' else False
        result = sorted(result, key=lambda k: k[key], reverse=reverse)

    return result


def set_default_figure():
    """
    設定繪圖預設值
    """

    # 繪布設定
    plt.rcParams['figure.figsize'] = (18, 21)
    sns.set(rc={'figure.figsize':(18, 21)})
    if os.sys.platform == 'darwin':
        plt.rcParams['font.family'] = "Heiti TC"
        plt.rcParams['font.sans-serif'] = "Heiti TC"
    elif os.sys.platform == 'linux':
        plt.rcParams['font.family'] = "Noto Sans CJK JP"
        plt.rcParams['font.sans-serif'] = "Noto Sans CJK JP"
    else:
        pass

    plt.rcParams['font.size'] = 16
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['grid.color'] = 'k'
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.linewidth'] = 0.5


def load_pickles(prefix: str, dir_name: str = "./pickle"):
    """
    載入指定目錄下符合指定前置字串的 pickle 檔案

    :param prerfix: pickle 檔名前置字串
    :param dir_name: pickle 所在目錄

    :return: pickles: 符合條件的 pickle 檔案內容的 dict
    """

    pickle_files = []
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file.startswith(prefix):
                pickle_files.append(os.path.join(root,file))

    pickles = {}

    for file in pickle_files:
        print(f"Loading pickle: {file}" )
        data = pickle.load(open(file, 'rb'))
        pickles[os.path.basename(file)] = data

    return pickles

# def load_classficatiton_reports(dir_name: str = './pickle'):
#     """
#     載入模組測試報告記錄檔

#     :param dir_name: 記錄檔所在目錄

#     :return: model_reports: 測試報告
#     """

#     pickle_files = []
#     for root, dirs, files in os.walk(dir_name):
#         for file in files:
#             if re.match('^classification_\d{8}.pickle$', file):
#                 pickle_files.append(os.path.join(root, file))
            
#     pickle_files.sort()  

#     model_reports = []
#     test_size = set()
#     for pickle_file in pickle_files:
#         print(f"Loading pickle: {pickle_file}" )
#         reports = pickle.load(open(pickle_file, 'rb'))
#         for report in reports:
#             test_size.add(report['test_size'])
#         print(test_size)
#         model_reports.extend(reports)    

#     return model_reports 