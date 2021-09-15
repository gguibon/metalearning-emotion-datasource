# Meta-learning for Emotion Classification with Unseen Datasource

Code for the ACL MetaNLP paper: [Meta-learning: Leveraging a Social Network Annotated Data Set for the Classification of Dialog Utterances into Previously Unseen Emotional Categories](https://aclanthology.org/2021.metanlp-1.9/)

Authors: Gaël Guibon, Matthieu Labeau, Hélène Flamein, Luce Lefeuvre, Chloé Clavel

## Citing

If you find this repo or paper useful, please cite the following paper:
```
@inproceedings{guibon-etal-2021-meta-learning,
    title = "Meta-learning for Classifying Previously Unseen Data Source into Previously Unseen Emotional Categories",
    author = {Guibon, Ga{\"e}l  and
      Labeau, Matthieu  and
      Flamein, H{\'e}l{\`e}ne  and
      Lefeuvre, Luce  and
      Clavel, Chlo{\'e}},
    booktitle = "Proceedings of the 1st Workshop on Meta Learning and Its Applications to Natural Language Processing",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.metanlp-1.9",
    doi = "10.18653/v1/2021.metanlp-1.9",
    pages = "76--89",
    abstract = "In this paper, we place ourselves in a classification scenario in which the target classes and data type are not accessible during training. We use a meta-learning approach to determine whether or not meta-trained information from common social network data with fine-grained emotion labels can achieve competitive performance on messages labeled with different emotion categories. We leverage few-shot learning to match with the classification scenario and consider metric learning based meta-learning by setting up Prototypical Networks with a Transformer encoder, trained in an episodic fashion. This approach proves to be effective for capturing meta-information from a source emotional tag set to predict previously unseen emotional tags. Even though shifting the data type triggers an expected performance drop, our meta-learning approach achieves decent results when compared to the fully supervised one.",
}
```

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
