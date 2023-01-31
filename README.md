# PBCNet

## model_code
The skeleton code of the PBCNet model.

## results_in_our_article
Summary of the outcome data reported in the article.

## PBCNet.pth
The trained PBCNet

## environment.txt
The packages required for PBCNet.

## data
Note: The nature of pairwise input required for PBCNet results in one sample appearing in multiple sample pairs. Therefore, to reduce the time spent on data processing during training and prediction, we store most of the data as pickle files.
#### 1. FEP1
The ligands in the FEP1 set on mol2 and sdf formats; the protein and pocket files on mol2 and pdb formats; and the computing results of intermolecular interactions.
#### 2. FEP2
The ligands in the FEP2 set on mol2 and sdf formats; the protein and pocket files on mol2 and pdb formats; and the computing results of intermolecular interactions.
#### 3. test_set_fep_graph_rmH_I
The corresponding pickle files of the FEP1 set.
#### 4. test_set_fep+_graph_rmH_I
The corresponding pickle files of the FEP2 set.
#### 5. finetune_input_files
The model input files (csv files) for finetune operation.
#### 6. ic50_graph_rmH_new
The corresponding pickle files of the Training set.
#### 7. mean_600000train_all_pair_withoutFEP.csv
The  model input files (csv files) for tarining.
#### 8. selection
The ligands in the selection test set on mol2 and sdf formats and the protein and pocket files on mol2 and pdb formats.
#### 9. selection_graph
The corresponding pickle files of the selection test set.

## How to train the PBCNet?
bash ./run_training.sh

## How to finetune the PBCNet?
bash ./run_finetune.sh
