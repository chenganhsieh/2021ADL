# ADL HW0 - R09944010
### Run all tasks
指令：
```python=
python3 main.py --data_dir 資料位置 --all 
```
### Preprocess
將train.csv 分成pos data 以及neg data，並分別做bag of words vector
最後得到positve vector size: negative vector size:
指令：
```python=
python3 main.py --preprocess
```
輸出positive.pkl 以及negative.pkl

### Training
利用preprocess所得到每筆data的bag of words vector(含pos 和 neg)
用linear model進行訓練，計算最終label所用的方法是將pos model以及neg model所機率去取平均
指令：
```python=
python3 main.py --train
```
輸出model_positive.pkl 以及model_negative.pkl 的model checkpoint
輸出answer_準確率.csv 為test的csv ，可放到kaggle做測試
### Predicting
須先執行training 取得model checkpoint
指令：
```python=
python3 main.py --predict
```
輸出 answer.csv