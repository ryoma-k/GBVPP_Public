# Google Brain - Ventilator Pressure Prediction 9th place solution

Discussion: https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/285353


## directory composition
```
|--README.md
|--requirements.txt
|--src
|--input
  |--ventilator-pressure-prediction
    |--sample_submission.csv
    |--test.csv
    |--train.csv
```

## training code
```
# at `src` directroy
# data preparation
python main.py -cp yamls/base.yaml yamls/preprocess.yaml

# run laplace likelihood optimization
python main.py -cp yamls/base.yaml

# run l1 optimization
python main.py -cp yamls/base.yaml yamls/l1.yaml
```

## tensorboard
```
tensorboard --logdir result
```

## learning curve
![image](https://user-images.githubusercontent.com/45588624/140611650-192b8eb6-4b72-4f78-bfa3-4ed35e845ac6.png)

## note

sorry if it's hard to read the codes :)
I override `src/base` with `src/lstm`, so codes for this competition are in `src/lstm` directory.

I found dropout in lstm stablizes the training. You can add it with `python main.py -cp yamls/base.yaml yamls/dropout.yaml`
