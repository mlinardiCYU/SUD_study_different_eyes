# Studying SUD Through Different Eyes
**Bruno Machado Carneiro, Michele Linardi, Julien Longhi.**

Repository of the paper: "Studying Socially Unacceptable Discourse Classification (SUD) through different eyes: Are we on the same page ?"
It contains the source code of the implemented SUD classification models and the analyzed corpora.
The paper appears in the proceesings of the International Conference on CMC and Social Media Corpora for the Humanities.

## Data

Details on the data acquisition and preprocessing are described [here](data/data.md).


## Usage  

### Training  

```
train.py [--data data_path] 
         [--mode {binary, multi-class}]
         [--batch_size batch_size]
         [--epochs epochs_count]
         [--lr learning_rate]
         [--smoothing smoothing_rate]
```

* **Arguments:**
```
--data          Path to the dataset
--mode          Perform binary or multi-class SUD classification
--batch_size    Batch size
--epochs        Number of epochs
--lr            Learning rate
--smoothing     Label smoothing rate
```

### Testing  

```
test.py [--data data_path] 
        [--model model_path]
        [--mode {binary, multi-class}]
```

* **Arguments:**
```
--data          Path to the dataset
--model         Path to the trained model
--mode          Perform binary or multi-class SUD classification
```