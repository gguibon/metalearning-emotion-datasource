# Meta-learning: Leveraging a Social Network Annotated Data Set for the Classification of Dialog Utterances into Previously Unseen Emotional Categories

Gaël Guibon, Matthieu Labeau, Hélène Flamein, Luce Lefeuvre, Chloé Clavel

## Reproducibility

To reproduce results please run the main python script as follows.


### Display options:
```
python3 metalearning.py --help
```

### Reproduce our best results using meta learning on emotion labels and data sources:
```
python3 metalearning.py --encoder transfo
```

### Compare them with Bao et al. Metalearning wih Ridge Regressor:
```
python3 metalearning.py --encoder meta
```


### Reproduce supervised results on DailyDialog splits:
```
python3 metalearning.py --task supervised_dailydialog --encoder transfo
```

### Reproduce suggested supervised experiment:

```
python3 metalearning.py --task supervised_goemotions_on_dailydialog --encoder transfo
```


## Environnement

Please use the `requirements.txt` to install dependencies.
```
pip3 install -r requirements.txt
```

FastText embeddings are available [here](https://fasttext.cc/docs/en/english-vectors.html). Please put the `wiki-news-300d-1M.vec` file into the `data` directory.
