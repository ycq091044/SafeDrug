import dill
import numpy as np
import pandas as pd
import requests
import re

# fix mismatch between two mappings
def fix_mismatch(idx2atc, atc2ndc, ndc2atc_original_path):
    ndc2atc = pd.read_csv(open(ndc2atc_original_path, 'rb'))
    ndc2atc.ATC4 = ndc2atc.ATC4.apply(lambda x: x[:4])

    mismatch = []
    for k, v in idx2atc.items():
        if v in atc2ndc.NDC.tolist():
            pass
        else:
            mismatch.append(v)

    for i in mismatch:
        atc2ndc = atc2ndc.append({'NDC': i, 'NDC_orig': [s.replace('-', '') for s in ndc2atc[ndc2atc.ATC4 == i].NDC.tolist()]}, ignore_index=True)
        
    atc2ndc = atc2ndc.append({'NDC': 'seperator', 'NDC_orig': []}, ignore_index=True)
    atc2ndc = atc2ndc.append({'NDC': 'decoder_point', 'NDC_orig': []}, ignore_index=True)

    return atc2ndc

def ndc2smiles(NDC):
    url3 = 'https://ndclist.com/?s=' + NDC
    r3 = requests.get(url3)
    name = re.findall('<td data-title="Proprietary Name">(.+?)</td>', r3.text)[0]
    
    url = 'https://dev.drugbankplus.com/guides/tutorials/api_request?request_path=us/product_concepts?q=' + name
    r = requests.get(url)
    drugbankID = re.findall('(DB\d+)', r.text)[0]

    # re matching might need to update (drugbank may change their html script)
    url2 = 'https://www.drugbank.ca/drugs/' + drugbankID
    r2 = requests.get(url2)
    SMILES = re.findall('SMILES</dt><dd class="col-xl-10 col-md-9 col-sm-8"><div class="wrap">(.+?)</div>', r2.text)[0]
    return SMILES

def atc2smiles(atc2ndc):
    atc2SMILES = {}
    for k, ndc in atc2ndc.values:
        if k not in list(atc2SMILES.keys()):
            for index, code in enumerate(ndc):
                if index > 100: break
                try:
                    SMILES = ndc2smiles(code)
                    if 'href' in SMILES:
                        continue
                    print (k, index, len(ndc), SMILES)
                    if k not in atc2SMILES:
                        atc2SMILES[k] = set()
                    atc2SMILES[k].add(SMILES)
                    # if len(atc2SMILES[k]) >= 3:
                    #     break
                except:
                    pass
    return atc2SMILES


def idx2smiles(idx2atc, atc2SMILES):
    idx2drug = {}
    idx2drug['seperator'] = {}
    idx2drug['decoder_point'] = {}

    for idx, atc in idx2atc.items():
        try:
            idx2drug[idx] = atc2SMILES[atc]
        except:
            pass
    dill.dump(idx2drug, open('idx2drug.pkl', 'wb'))


if __name__ == '__main__':
    # get idx2atc
    path = './voc_final.pkl'
    voc_final = dill.load(open(path, 'rb'))
    idx2atc = voc_final['med_voc'].idx2word

    # get atc2ndc
    path = './ndc2drug.pkl'
    atc2ndc = dill.load(open(path, 'rb'))

    # fix atc2ndc mismatch
    ndc2atc_original_path = './ndc2atc_level4.csv'
    atc2ndc = fix_mismatch(idx2atc, atc2ndc, ndc2atc_original_path)

    # atc2smiles
    atc2SMILES = atc2smiles(atc2ndc)

    # idx2smiles (dumpped)
    idx2smiles(idx2atc, atc2SMILES)




