## 程式清單

| 程式名稱                  | 用途說明                                                                         |
| ------------------------- | -------------------------------------------------------------------------------- |
| base-all.py               | 建立所有模型在預設條件下的學習成效基準                                           |
| base-model.py             | 建立指定模型在預設條件下的學習成效基準（分散、補缺漏模型基準時使用）             |
| EDA.ipynb                 | 基礎資料探索                                                                     |
| evaluate-base.ipynb       | 取得所有模型在預設條件下的參考基準                                               |
| gridsearch_for_mlp.py     | 對 MLP 模型以不同參數進行交叉驗證以便取得較佳神經元參數                          |
| evaluate-gridsearch.ipynb | 視覺化 MLP 模型交叉驗證成果                                                      |
| exp-scenario.py           | 單一場景資料在指定模型下以不同座標系統及不同訓練資料量進行機器學習並記錄學習成效 |
| evaluate.ipynb            | 以視覺化方式從不同角度檢視各模型學習成效                                         |
| check-mlp-weight.py       | 驗證 MLP 模型在初始權重不同時會導致學習成效差異                                  |
| evaluate-mlp-weight.ipynb | 檢視 MLP 模型在不同初始權重下的學習成效                                          |
| utils.py                  | 通用工具程式                                                                     |
| pickle/*                  | 訓練結果|
| data/*                    | EHUCOUNT Dataset，請由 [EHUCOUNT Dataset: WiFi CSI measurements employed for a device-free people counting framework](https://www.researchgate.net/publication/332143935_EHUCOUNT_Dataset_WiFi_CSI_measurements_employed_for_a_device-free_people_counting_framework) 下載取得後將解壓縮所得的 6 個 Scenario_?.mat 檔直接存放於目錄下即可 |

