from xml.dom.pulldom import ErrorHandler
import pandas as pd
import dill
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import BRICS

##### process medications #####
# load med data
def med_process(med_file):
    med_pd = pd.read_csv(med_file, dtype={'NDC':'category'})

    # med_pd.drop(columns=['ROW_ID','DRUG_TYPE','DRUG_NAME_POE','DRUG_NAME_GENERIC',
    #                     'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
    #                     'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
    #                     'ROUTE','ENDDATE','DRUG'], axis=1, inplace=True)
    med_pd.drop(columns=['ROW_ID','DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC',
                        'FORMULARY_DRUG_CD','PROD_STRENGTH','DOSE_VAL_RX',
                        'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP', 'GSN', 'FORM_UNIT_DISP',
                        'ROUTE','ENDDATE'], axis=1, inplace=True)
    med_pd.drop(index = med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')    
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop(columns=['ICUSTAY_ID'])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    return med_pd

# ATC3-to-drugname
def ATC3toDrug(med_pd):
    atc3toDrugDict = {}
    for atc3, drugname in med_pd[['ATC3', 'DRUG']].values:
        if atc3 in atc3toDrugDict:
            atc3toDrugDict[atc3].add(drugname)
        else:
            atc3toDrugDict[atc3] = set(drugname)

    return atc3toDrugDict

def atc3toSMILES(ATC3toDrugDict, druginfo):
    drug2smiles = {}
    atc3tosmiles = {}
    for drugname, smiles in druginfo[['name', 'moldb_smiles']].values:
        if type(smiles) == type('a'):
            drug2smiles[drugname] = smiles
    for atc3, drug in ATC3toDrugDict.items():
        temp = []
        for d in drug:
            try:
                temp.append(drug2smiles[d])
            except:
                pass
        if len(temp) > 0:
            atc3tosmiles[atc3] = temp[:3]
    
    return atc3tosmiles

# medication mapping
def codeMapping2atc4(med_pd):
    with open(rxnorm2RXCUI_file, 'r') as f:
        rxnorm2RXCUI = eval(f.read())
    med_pd['RXCUI'] = med_pd['NDC'].map(rxnorm2RXCUI)
    med_pd.dropna(inplace=True)

    rxnorm2atc4 = pd.read_csv(RXCUI2atc4_file)
    rxnorm2atc4 = rxnorm2atc4.drop(columns=['YEAR','MONTH','NDC'])
    rxnorm2atc4.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index = med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)
    
    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc4, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd['ATC4'] = med_pd['ATC4'].map(lambda x: x[:4])
    med_pd = med_pd.rename(columns={'ATC4':'ATC3'})
    med_pd = med_pd.drop_duplicates()    
    med_pd = med_pd.reset_index(drop=True)
    return med_pd

# visit >= 2
def process_visit_lg2(med_pd):
    a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')['HADM_ID'].unique().reset_index()
    a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x:len(x))
    a = a[a['HADM_ID_Len'] > 1]
    return a 

# most common medications
def filter_300_most_med(med_pd):
    med_count = med_pd.groupby(by=['ATC3']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    med_pd = med_pd[med_pd['ATC3'].isin(med_count.loc[:299, 'ATC3'])]
    
    return med_pd.reset_index(drop=True)

##### process diagnosis #####
def diag_process(diag_file):
    diag_pd = pd.read_csv(diag_file)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM','ROW_ID'],inplace=True)
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID','HADM_ID'], inplace=True)
    diag_pd = diag_pd.reset_index(drop=True)

    def filter_2000_most_diag(diag_pd):
        diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
        diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]
        
        return diag_pd.reset_index(drop=True)

    diag_pd = filter_2000_most_diag(diag_pd)

    return diag_pd

##### process procedure #####
def procedure_process(procedure_file):
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE':'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd

def filter_1000_most_pro(pro_pd):
    pro_count = pro_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    pro_pd = pro_pd[pro_pd['ICD9_CODE'].isin(pro_count.loc[:1000, 'ICD9_CODE'])]
    
    return pro_pd.reset_index(drop=True) 

###### combine three tables #####
def combine_process(med_pd, diag_pd, pro_pd):

    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    # flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index()  
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ATC3'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})  
    med_pd['ATC3'] = med_pd['ATC3'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    #     data['ICD9_CODE_Len'] = data['ICD9_CODE'].map(lambda x: len(x))
    data['ATC3_num'] = data['ATC3'].map(lambda x: len(x))

    return data

def statistics(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))
    
    diag = data['ICD9_CODE'].values
    med = data['ATC3'].values
    pro = data['PRO_CODE'].values
    
    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])
    
    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))
    
    avg_diag, avg_med, avg_pro, max_diag, max_med, max_pro, cnt, max_visit, avg_visit = [0 for i in range(9)]

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x, y, z = [], [], []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['ATC3']))
            z.extend(list(row['PRO_CODE']))
        x, y, z = set(x), set(y), set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y) 
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt
        
    print('#avg of diagnoses ', avg_diag/ cnt)
    print('#avg of medicines ', avg_med/ cnt)
    print('#avg of procedures ', avg_pro/ cnt)
    print('#avg of vists ', avg_visit/ len(data['SUBJECT_ID'].unique()))
    
    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)

##### indexing file and final record
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)
                
# create voc set
def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()
    
    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['ATC3'])
        pro_voc.add_sentence(row['PRO_CODE'])
    
    dill.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc ,'pro_voc':pro_voc}, file=open(vocabulary_file,'wb'))
    return diag_voc, med_voc, pro_voc

# create final records
def create_patient_record(df, diag_voc, med_voc, pro_voc):
    records = [] # (patient, code_kind:3, codes)  code_kind:diag, proc, med
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['ATC3']])
            patient.append(admission)
        records.append(patient) 
    dill.dump(obj=records, file=open(ehr_sequence_file, 'wb'))
    return records
        
# get ddi matrix
def get_ddi_matrix(records, med_voc, ddi_file):

    TOPK = 40 # topk drug-drug interaction
    cid2atc_dic = defaultdict(set)
    med_voc_size = len(med_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]
    atc3_atc4_dic = defaultdict(set)
    for item in med_unique_word:
        atc3_atc4_dic[item[:4]].add(item)
    
    with open(cid2atc6_file, 'r') as f:
        for line in f:
            line_ls = line[:-1].split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc3_atc4_dic[atc[:4]]) != 0:
                    cid2atc_dic[cid].add(atc[:4])
            
    # ddi load
    ddi_df = pd.read_csv(ddi_file)
    # fliter sever side effect 
    ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name'])\
        .size().reset_index().rename(columns={0:'count'}).sort_values(by=['count'],ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-TOPK:,:]
    # ddi_most_pd = pd.DataFrame(columns=['Side Effect Name'], data=['as','asd','as'])
    fliter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1','STITCH 2']].drop_duplicates().reset_index(drop=True)

    # weighted ehr adj 
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j<=i:
                        continue
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1
    dill.dump(ehr_adj, open(ehr_adjacency_file, 'wb'))  

    # ddi adj
    ddi_adj = np.zeros((med_voc_size,med_voc_size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row['STITCH 1']
        cid2 = row['STITCH 2']
        
        # cid -> atc_level3
        for atc_i in cid2atc_dic[cid1]:
            for atc_j in cid2atc_dic[cid2]:
                
                # atc_level3 -> atc_level4
                for i in atc3_atc4_dic[atc_i]:
                    for j in atc3_atc4_dic[atc_j]:
                        if med_voc.word2idx[i] != med_voc.word2idx[j]:
                            ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                            ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1
    dill.dump(ddi_adj, open(ddi_adjacency_file, 'wb')) 

    return ddi_adj

def get_ddi_mask(atc42SMLES, med_voc):

    # ATC3_List[22] = {0}
    # ATC3_List[25] = {0}
    # ATC3_List[27] = {0}
    fraction = []
    for k, v in med_voc.idx2word.items():
        tempF = set()
        for SMILES in atc42SMLES[v]:
            try:
                m = BRICS.BRICSDecompose(Chem.MolFromSmiles(SMILES))
                for frac in m:
                    tempF.add(frac)
            except:
                pass
        fraction.append(tempF)
    fracSet = []
    for i in fraction:
        fracSet += i
    fracSet = list(set(fracSet)) # set of all segments

    ddi_matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))
    for i, fracList in enumerate(fraction):
        for frac in fracList:
            ddi_matrix[i, fracSet.index(frac)] = 1
    return ddi_matrix


if __name__ == '__main__':

    # files can be downloaded from https://mimic.physionet.org/gettingstarted/dbsetup/
    # please change into your own MIMIC folder
    med_file = '/srv/local/data/physionet.org/files/mimiciii/1.4/PRESCRIPTIONS.csv'
    diag_file = '/srv/local/data/physionet.org/files/mimiciii/1.4/DIAGNOSES_ICD.csv'
    procedure_file = '/srv/local/data/physionet.org/files/mimiciii/1.4/PROCEDURES_ICD.csv'

    # input auxiliary files
    med_structure_file = './output/atc32SMILES.pkl'
    RXCUI2atc4_file = './input/RXCUI2atc4.csv' 
    cid2atc6_file = './input/drug-atc.csv'
    rxnorm2RXCUI_file = './input/rxnorm2RXCUI.txt'
    ddi_file = './input/drug-DDI.csv'
    drugbankinfo = './input/drugbank_drugs_info.csv'

    # output files
    ddi_adjacency_file = "./output/ddi_A_final.pkl"
    ehr_adjacency_file = "./output/ehr_adj_final.pkl"
    ehr_sequence_file = "./output/records_final.pkl"
    vocabulary_file = "./output/voc_final.pkl"
    ddi_mask_H_file = "./output/ddi_mask_H.pkl"
    atc3toSMILES_file = './output/atc3toSMILES.pkl'
    
    # for med
    med_pd = med_process(med_file)
    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)    
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner').reset_index(drop=True) 

    med_pd = codeMapping2atc4(med_pd)
    med_pd = filter_300_most_med(med_pd)

    # med to SMILES mapping
    atc3toDrug = ATC3toDrug(med_pd)
    druginfo = pd.read_csv(drugbankinfo)
    atc3toSMILES = atc3toSMILES(atc3toDrug, druginfo)
    dill.dump(atc3toSMILES, open(atc3toSMILES_file,'wb'))
    med_pd = med_pd[med_pd.ATC3.isin(atc3toSMILES.keys())]
    print ('complete medication processing')

    # for diagnosis
    diag_pd = diag_process(diag_file)

    print ('complete diagnosis processing')

    # for procedure
    pro_pd = procedure_process(procedure_file)
    # pro_pd = filter_1000_most_pro(pro_pd)

    print ('complete procedure processing')

    # combine
    data = combine_process(med_pd, diag_pd, pro_pd)
    statistics(data)
    print ('complete combining')

    # create vocab
    diag_voc, med_voc, pro_voc = create_str_token_mapping(data)
    print ("obtain voc")

    # create ehr sequence data
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)
    print ("obtain ehr sequence data")

    # create ddi adj matrix
    ddi_adj = get_ddi_matrix(records, med_voc, ddi_file)
    print ("obtain ddi adj matrix")

    # get ddi_mask_H
    ddi_mask_H = get_ddi_mask(atc3toSMILES, med_voc)
    dill.dump(ddi_mask_H, open(ddi_mask_H_file, 'wb'))
