# NN parametrization of DFT functionals


## Preparing data for parametrization


1) Create and activate virtual environment, install packages from requirements.txt, download the dataset following the link in  MN_dataset folder
2) Create /*data* folder in the train_models directory, load the *.h5* files in it
3) Run `prepare_data.py` to split the dataset into train and validation and save it in ./*checkpoints*

## Parametrization
To train and validate the models, run `calculations.py` with suitable parameters.



