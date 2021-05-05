from rdkit import Chem
from rdkit.Chem import BRICS
import dill
import numpy as np
# from pinky.smiles import smilin
# from pinky.fingerprints import ecfp

NDCList = dill.load(open('./idx2drug.pkl', 'rb'))
voc = dill.load(open('./voc_final.pkl', 'rb'))
med_voc = voc['med_voc']

"""
fraction = []
for k, v in med_voc.idx2word.items():
    tempF = []
    for SMILES in NDCList[v]:
        try:
            m = ecfp(smilin(SMILES), radius=2)
            tempF.append(m)
        except:
            pass
    fraction.append(tempF)


"""
NDCList[22] = {0}
NDCList[25] = {0}
NDCList[27] = {0}
fraction = []
for k, v in med_voc.idx2word.items():
    tempF = set()
    for SMILES in NDCList[k]:
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
fracSet = list(set(fracSet))

ddi_matrix = np.zeros((len(med_voc.idx2word), len(fracSet)))

for i, fracList in enumerate(fraction):
    for frac in fracList:
        ddi_matrix[i, fracSet.index(frac)] = 1

dill.dump(ddi_matrix, open('ddi_mask_H.pkl', 'wb'))
