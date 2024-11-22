[Original Readme](./README_original.md)

## Train Model
The main configurations are defined in `config.yml`. To change the model type for training, edit the following parameter

``` yaml
model:
  configuration: "ggnn" # can also be "great", "rnn", etc
training:
  max_steps: 60 # number of dev steps duing training
```

Then, use the following commend to fire the training

``` bash
python running/run_model.py <path_to_dataset> vocab.txt <config_file_path> -l <log_path>
```

Change `path_to_dataset` to be the path of the dataset
Change `config_file_path` to be the path of the config file
Change `log_path` to be the path of the log file (need to be used later during evaluation)

## Training With Type Checking and Random Filtering

First, download the typecheck information, random filtering information and updated training dataset from [here](https://drive.google.com/file/d/1t1Qv9T4ltTr01yuWIjLwLRBVIDvFmWaQ/view?usp=sharing) and decompress them to the path containing the original dataset (Overwrite the original training samples with the updated training samples). The updated training dataset contains unique identifier for each data sample and this is necessary due to the data loading strategy of tensorflow. 

To run training with typechecking, change the following parameter to `true`

``` yaml
data:
    typechecking: true
    typechecking_oversample: true
```

**Notice that these two parameters should not be `true` at the same time. This will filter the training data by both type checking and random filtering result and it is likely not what you want.**

## Evaluate
To evaluate the model, use the following command

``` bash
python running/run_model.py <data_path> vocab.txt <config_file_path> -m <model_folder> -l <log_path> -e True
```


The `data_path`, `config_file_path`, `log_path` should be the same as the training command. The `model_folder` should be the path of the model folder that contains model checkpoints from training.