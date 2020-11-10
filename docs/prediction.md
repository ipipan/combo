# Prediction

## ConLLU file prediction:
Input and output are both in `*.conllu` format.
```bash
combo --mode predict --model_path your_model_tar_gz --input_file your_conllu_file --output_file your_output_file --silent
```

## Console
Works for models where input was text-based only.

Interactive testing in console (load model and just type sentence in console).

```bash
combo --mode predict --model_path your_model_tar_gz --input_file "-" --nosilent
```
## Raw text
Works for models where input was text-based only. 

Input: one sentence per line.

Output: List of token jsons.

```bash
combo --mode predict --model_path your_model_tar_gz --input_file your_text_file --output_file your_output_file --silent --noconllu_format
```
### Advanced

There are 2 tokenizers: whitespace and spacy-based (`en_core_web_sm` model).

Use either `--predictor_name semantic-multitask-predictor` or `--predictor_name semantic-multitask-predictor-spacy`.

## Python
```python
import combo.predict as predict

model_path = "your_model.tar.gz"
nlp = predict.SemanticMultitaskPredictor.from_pretrained(model_path)
sentence = nlp("Sentence to parse.")
```
