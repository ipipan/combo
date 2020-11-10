# COMBO
<p align="center">
    A language-independent NLP system for dependency parsing, part-of-speech tagging, lemmatisation and more built on top of PyTorch and AllenNLP.
</p>
<hr/>
<p align="center">
    <a href="https://github.com/ipipan/combo/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/ipipan/combo.svg?color=blue&cachedrop">
    </a>
</p>

## Quick start
Clone this repository and install COMBO (we suggest using virtualenv/conda with Python 3.6+):
```bash
git clone https://github.com/ipipan/combo.git
cd combo
python setup.py develop
```
Run the following lines in your Python console to make predictions with a pre-trained model:
```python
import combo.predict as predict

nlp = predict.SemanticMultitaskPredictor.from_pretrained("polish-herbert-base")
sentence = nlp("Moje zdanie.")
print(sentence.tokens)
```

## Details

- [**Installation**](docs/installation.md)
- [**Pre-trained models**](docs/models.md)
- [**Training**](docs/training.md)
- [**Prediction**](docs/prediction.md)

