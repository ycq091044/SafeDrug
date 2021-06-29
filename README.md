# Data and Code for IJCAI'21 paper - SafeDrug

### folder specification
- data/
    - mapping files that collected from external sources
        - This is the data folder
        - drug-atc.csv: drug to atc code mapping file
        - drug-DDI.csv: this a large file, could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
        - ndc2atc_level4.csv: NDC code to ATC-4 code mapping file
        - ndc2xnorm_mapping.txt: NDC to xnorm mapping file
        - id2drug.pkl: drug ID to drug SMILES string dict
    - other files that generated from mapping files and MIMIC dataset (we attach these files here, user could use our provided scriots to generate)
        - data_final.pkl: intermediate result
        - ddi_A_final.pkl: ddi matrix
        - ddi_matrix.pkl: H mask structure
        - ehr_adj_final.pkl: used in GAMENet baseline
        - idx2ndc.pkl: idx2ndc mapping file
        - ndc2drug.pkl: ndc2drug mapping file
        - (important) records_final.pkl: 100 patient visit-level record samples. Under MIMIC Dataset policy, we are not allowed to distribute the datasets. Practioners could go to https://mimic.physionet.org/about/mimic/ and requrest the access to MIMIC-III dataset and then run our processing script to get the complete preprocessed dataset file.
        - voc_final.pkl: diag/prod/med dictionary
    - dataset processing scripts
        - processing.py: is used to process the MIMIC original dataset
        - ddi_mask_H.py: is used to get ddi_mask_H.pkl
        - get_SMILES.py: is our crawler, used to transform atc-4 code to SMILES string. It generates idx2drug.pkl.
- src/
    - SafeDrug.py: our model
    - Epoch_49_TARGET_0.06_JA_0.5183_DDI_0.05854.model: the model we trained on the training set
    - baselines:
        - GAMENet.py
        - DMNC.py
        - Leap.py
        - Retain.py
        - ECC.py
        - LR.py
    - setting file
        - model.py
        - util.py
        - layer.py

### dependency
```python
conda create -c conda-forge -n SafeDrug  rdkit # install rdkit env, you can change "SafeDrug" to [Your own env name]
pip install scikit-learn, torch, dill, dnc
pip install [xxx] # any required package if necessary, maybe do not specify the version, the packages should be compatible with rdkit
```

### argument

    usage: SafeDrug.py [-h] [--Test] [--model_name MODEL_NAME]
                   [--resume_path RESUME_PATH] [--lr LR]
                   [--target_ddi TARGET_DDI] [--kp KP] [--dim DIM]

    optional arguments:
      -h, --help            show this help message and exit
      --Test                test mode
      --model_name MODEL_NAME
                            model name
      --resume_path RESUME_PATH
                            resume path
      --lr LR               learning rate
      --target_ddi TARGET_DDI
                            target ddi
      --kp KP               coefficient of P signal
      --dim DIM             dimension

### run the code
run ```python SafeDrug.py```. If you cannot run the code on GPU, just change line 101, "cuda" to "cpu".

### cite
```bibtex
@inproceedings{yang2021safedrug,
    title = {SafeDrug: Dual Molecular Graph Encoders for Safe Drug Recommendations},
    author = {Yang, Chaoqi and Xiao, Cao and Ma, Fenglong and Glass, Lucas and Sun, Jimeng},
    booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2021},
    year = {2021}
}
```

Welcome to contact me <chaoqiy2@illinois.edu> for any question.
