# data and code for IJCAI21 submission - SafeDrug

### Reproduce the IJCAI paper results
- Step 1: download the DDI file and move it to the data folder
    - https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
- Step 2: processing the data to get a complete records_final.pkl
    ```python
    # modify line 294-296 to your own path
    med_file = './physionet.org/files/mimiciii/1.4/PRESCRIPTIONS.csv'
    diag_file = './physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv'
    procedure_file = './physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv'
    ```
    ```
    python processing.py
    ```
- Step 3: run model
    ```
    python SafeDrug.py
    ```

### Folder specification
- data/
    - mapping files that collected from external sources
        - This is the data folder
        - drug-atc.csv: drug to atc code mapping file
        - drug-DDI.csv: this a large file, could be downloaded from https://www.dropbox.com/s/8os4pd2zmp2jemd/drug-DDI.csv?dl=0
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

### Dependency
python 3.7, scipy 1.5.2, pandas 1.1.3, torch 1.4.0, numpy 1.19.2, dill, rdkit (installation refer to https://www.rdkit.org/docs/Install.html)


### Cite
```bibtex
@inproceedings{yang2021safedrug,
    title = {SafeDrug: Dual Molecular Graph Encoders for Safe Drug Recommendations},
    author = {Chaoqi Yang, Cao Xiao, Fenglong Ma, Lucas Glass and Jimeng Sun},
    booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2021},
    year = {2021}
}
```
