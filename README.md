# Data and Code for IJCAI'21 paper - SafeDrug

### Folder Specification
- ```data/```
    - **procesing.py** our data preprocessing file.
    - ```Input/``` (extracted from external resources)
        - **PRESCRIPTIONS.csv**: the prescription file from MIMIC-III raw dataset
        - **DIAGNOSES_ICD.csv**: the diagnosis file from MIMIC-III raw dataset
        - **PROCEDURES_ICD.csv**: the procedure file from MIMIC-III raw dataset
        - **RXCUI2atc4.csv**: this is a NDC-RXCUI-ATC4 mapping file, and we only need the RXCUI to ATC4 mapping. This file is obtained from https://github.com/sjy1203/GAMENet, where the name is called ndc2atc_level4.csv.
        - **drug-atc.csv**: this is a CID-ATC file, which gives the mapping from CID code to detailed ATC code (we will use the prefix of the ATC code latter for aggregation). This file is obtained from https://github.com/sjy1203/GAMENet.
        - **rxnorm2RXCUI.txt**: rxnorm to RXCUI mapping file. This file is obtained from https://github.com/sjy1203/GAMENet, where the name is called ndc2rxnorm_mapping.csv.
        - **drugbank_drugs_info.csv**: drug information table downloaded from drugbank here https://www.dropbox.com/s/angoirabxurjljh/drugbank_drugs_info.csv?dl=0, which is used to map drug name to drug SMILES string.
        - **drug-DDI.csv**: this a large file, containing the drug DDI information, coded by CID. The file could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
    - ```Output/```
        - **atc3toSMILES.pkl**: drug ID (we use ATC-3 level code to represent drug ID) to drug SMILES string dict
        - **ddi_A_final.pkl**: ddi adjacency matrix
        - **ddi_matrix_H.pkl**: H mask structure (This file is created by **ddi_mask_H.py**)
        - **ehr_adj_final.pkl****: used in GAMENet baseline (if two drugs appear in one set, then they are connected)
        - **records_final.pkl**: The final diagnosis-procedure-medication EHR records of each patient, used for train/val/test split.
        - **voc_final.pkl**: diag/prod/med index to code dictionary
- ```src/```
    - **SafeDrug.py**: our model
    - baselines:
        - **GAMENet.py**
        - **DMNC.py**: there are some issues for the latest dnc package, please refer to the original DMNC repo https://github.com/thaihungle/DMNC
        - **Leap.py**
        - **Retain.py**
        - **ECC.py**
        - **LR.py**
    - setting file
        - **model.py**
        - **util.py**
        - **layer.py**

> Note that **./data/get_SMILES.py [NOT DIRECTLY USED NOW]** is the previous crawler, given the drug ATC-3 level code (four digit, e.g., 'A01A'), our crawler returns (a set of) SMILES strings of that ATC-3 class (crawled from drugbank). This file needs atc2rxnorm.pkl (which maps ATC-3 code to rxnorm code and then query to drugbank), generated from rxnorm2RXCUI.txt and RXCUI2atc4.csv. However, due to the structure change of drugbank, it is not used in the current pipeline.

> Now, we are using **drugbank_drugs_info.csv** to obtain the SMILES string for each ATC3 code (previously we use get_SMILES.py), thus, the data statistics change a bit. The current statistics are shown below:

```
#patients  6350
#clinical events  15032
#diagnosis  1958
#med  112
#procedure 1430
#avg of diagnoses  10.5089143161256
#avg of medicines  11.647751463544438
#avg of procedures  3.8436668440659925
#avg of vists  2.367244094488189
#max of diagnoses  128
#max of medicines  64
#max of procedures  50
#max of visit  29
```
### High-level Clarifications on How to Map ATC Code to SMILES
- The original **PRESCRIPTIONS.csv** file provides ```rxnorm->drugname``` mapping (the ```rxnorm``` value is indicated in ```NDC``` column)
- Use the **rxnorm2RXCUI.txt** file for ```rxnorm->RXCUI``` mapping (now we have ```RXCUI->drugname```)
  - in https://github.com/ycq091044/SafeDrug/blob/main/data/processing.py#70
- Use the **RXCUI2atc4.csv** file for ```RXCUI->atc4``` mapping, then change ```atc4``` to ```atc3``` (now we have ```atc3->drugname```)
  - in https://github.com/ycq091044/SafeDrug/blob/main/data/processing.py#80
- Use the **drugbank_drugs_info.csv** file for ```drug->SMILES``` mapping (now we have ```atc3->SMILES```)
  - in https://github.com/ycq091044/SafeDrug/blob/main/data/processing.py#48
- ```atc3``` is a coarse-granular drug classification, one ```atc3``` code contains multiple SMILES strings.

### Step 1: Package Dependency

- first, install the rdkit conda environment
```python
conda create -c conda-forge -n SafeDrug  rdkit
conda activate SafeDrug
```

- then, in SafeDrug environment, install the following package
```python
pip install scikit-learn, dill, dnc
```
Note that torch setup may vary according to GPU hardware. Generally, run the following
```python
pip install torch
```
If you are using RTX 3090, then plase use the following, which is the right way to make torch work.
```python
python3 -m pip install --user torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- Finally, install other packages if necessary
```python
pip install [xxx] # any required package if necessary, maybe do not specify the version, the packages should be compatible with rdkit
```

Here is a list of reference versions for all package

```shell
pandas: 1.3.0
dill: 0.3.4
torch: 1.8.0+cu111
rdkit: 2021.03.4
scikit-learn: 0.24.2
numpy: 1.21.1
```

Let us know any of the package dependency issue. Please pay special attention to pandas, some report that a high version of pandas would raise error for dill loading.


### Step 2: Data Processing

- Go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset (You may need to get the certificate)

  ```python
  cd ./data
  wget -r -N -c -np --user [account] --ask-password https://physionet.org/files/mimiciii/1.4/
  ```

- go into the folder and unzip three main files

  ```python
  cd ./physionet.org/files/mimiciii/1.4
  gzip -d PROCEDURES_ICD.csv.gz # procedure information
  gzip -d PRESCRIPTIONS.csv.gz  # prescription information
  gzip -d DIAGNOSES_ICD.csv.gz  # diagnosis information
  ```

- download the DDI file and move it to the data folder
  download https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
  ```python
  mv drug-DDI.csv ./data
  ```

- processing the data to get a complete records_final.pkl

  ```python
  cd ./data
  vim processing.py
  
  # line 323-325
  # med_file = './physionet.org/files/mimiciii/1.4/PRESCRIPTIONS.csv'
  # diag_file = './physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv'
  # procedure_file = './physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv'
  
  python processing.py
  ```


### Step 3: run the code

```python
python SafeDrug.py
```

here is the argument:

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


### Citation
```bibtex
@inproceedings{yang2021safedrug,
    title = {SafeDrug: Dual Molecular Graph Encoders for Safe Drug Recommendations},
    author = {Yang, Chaoqi and Xiao, Cao and Ma, Fenglong and Glass, Lucas and Sun, Jimeng},
    booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2021},
    year = {2021}
}
```

Welcome to contact me <chaoqiy2@illinois.edu> for any question. Partial credit to https://github.com/sjy1203/GAMENet.
