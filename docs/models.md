# Models

Pre-trained models are available [here](http://mozart.ipipan.waw.pl/~mklimaszewski/models/).

## Automatic download
Python `from_pretrained` method will download the pre-trained model if the provided name (without the extension .tar.gz) matches one of the names in [here](http://mozart.ipipan.waw.pl/~mklimaszewski/models/).
```python
import combo.predict as predict

nlp = predict.SemanticMultitaskPredictor.from_pretrained("polish-herbert-base")
```
Otherwise it looks for a model in local env.

## Console prediction/Local model
If you want to use the console version of COMBO, you need to download a pre-trained model manually
```bash
wget http://mozart.ipipan.waw.pl/~mklimaszewski/models/polish-herbert-base.tar.gz
```
and pass it as a parameter (see [prediction doc](prediction.md)).
