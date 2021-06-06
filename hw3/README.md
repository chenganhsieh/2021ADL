# Homework 3 ADL NTU 109 Spring - R09944010

## Directory
./code/bestmodel: my best model checkpoint
./code/cache: cache file 
./code/logs: tensorboard log file and reproduced model

# Reproducibility
After the training, you have to do the following steps:
1. edit ./code/main.py:  args.predict_model to "./logs/default/version_*/YOUR_BEST_MODEL"
2. Run run.sh to predict the result


## Download
```shell
bash ./download.sh
```

## Train
This will execute all my training scripts.
```shell
bash train.sh /path/to/train.jsonl /path/to/public.jsonl
```

## Predict
```shell
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```


# Other detailed steps:

## Preprocess with policy gradient
```shell
python3.8 ./code/main.py --rl_ratio 0.3
```

## Plot graph
Use pytorch-lightning tensorBoardLogger
```shell
# After the training, execute this code:
python3.8 -m tensorboard.main --logdir= ./code/logs/default/version_*/YOUR_BEST_MODEL
```
My trainging graph:
./bestmode/events.out.tfevents.1622725969.cuda8.39006.0

