# Training

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
    combo --mode train --cuda_device 0
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
    combo --mode train --targets head,deprel
    ```

* use part-of-speech tags for predicting only dependency tree

    ```bash
    combo --mode train --targets head,deprel --features token,char,upostag
    ```
  
## Configuration

### Advanced
Config template [config.template.jsonnet](config.template.jsonnet) is formed in `allennlp` format so you can freely modify it.
There is configuration for all the training/model parameters (learning rates, epochs number etc.).
Some of them use `jsonnet` syntax to get values from configuration flags, however most of them can be modified directly there.