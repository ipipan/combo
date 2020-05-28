## Installation

Clone this repository and run:
```bash
python setup.py develop
```

## Training

Command:
```bash
combo --mode train \
      --training_data_path your_training_path \
      --validation_data_path your_validation_path
```

Options:
```bash
combo --helpfull
```

Examples (for clarity without training/validation data paths):

* train on gpu 0

    ```bash
    combo --mode train --cuda_davice 0
    ```

* use pretrained embeddings:

    ```bash
    combo --mode train --pretrained_tokens your_pretrained_embeddings_path --embedding_dim your_embeddings_dim
    ```

* use pretrained transformer embeddings:

    ```bash
    combo --mode train --pretrained_transformer_name your_choosen_pretrained_transformer
    ```

* predict only dependency tree:

    ```bash
    combo --mode train --targets head --targets deprel
    ```

* use part-of-speech tags for predicting only dependency tree

    ```bash
    combo --mode train --targets head --targets deprel --features token --features char --features upostag
    ```

Advanced configuration: [Configuration](#configuration)

## Prediction

### ConLLU file prediction:
Input and output are both in `*.conllu` format.
```bash
combo --mode predict --model_path your_model_tar_gz --input_file your_conllu_file --output_file your_output_file --silent
```

### Console
Works for models where input was text-based only.

Interactive testing in console (load model and just type sentence in console).

```bash
combo --mode predict --model_path your_model_tar_gz --input_file "-"
```
### Raw text
Works for models where input was text-based only. 

Input: one sentence per line.

Output: List of token jsons.

```bash
combo --mode predict --model_path your_model_tar_gz --input_file your_text_file --output_file your_output_file --silent
```
#### Advanced

There are 2 tokenizers: whitespace and spacy-based (`en_core_web_sm` model).

Use either `--predictor_name semantic-multitask-predictor` or `--predictor_name semantic-multitask-predictor-spacy`.

### Python
```python
import combo.predict as predict

model_path = "your_model.tar.gz"
predictor = predict.SemanticMultitaskPredictor.from_pretrained(model_path)
parsed_tree = predictor.predict_string("Sentence to parse.")["tree"]
```

## Configuration

### Advanced
Config template [config.template.jsonnet](config.template.jsonnet) is formed in `allennlp` format so you can freely modify it.
There is configuration for all the training/model parameters (learning rates, epochs number etc.).
Some of them use `jsonnet` syntax to get values from configuration flags, however most of them can be modified directly there.
