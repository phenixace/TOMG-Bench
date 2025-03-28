from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit.Chem import FragmentCatalog
from rdkit import DataStructs
from rdkit.Chem import AllChem
import os
import pandas as pd
import numpy as np
import torch


def smiles_to_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)

def calculate_similarity(smiles1, smiles2):
    fp1 = smiles_to_fingerprint(smiles1)
    fp2 = smiles_to_fingerprint(smiles2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# def calculate_novelty(new_smiles_list):
#     data = pd.read_csv("./data/sources/zinc250k/zinc250k_selfies.csv")
#     known_smiles_list = data["smiles"].tolist()
#     known_fps = np.array([smiles_to_fingerprint(smiles) for smiles in known_smiles_list])
#     new_fps = np.array([smiles_to_fingerprint(smiles) for smiles in new_smiles_list])
    
#     similarity_matrix = 1 - pairwise_distances(new_fps, known_fps, metric='jaccard')
    
#     max_similarities = np.max(similarity_matrix, axis=1)
#     novelties = 1 - max_similarities
    
#     return novelties

# def smiles_to_fingerprint(smiles, n_bits=2048):
#     mol = Chem.MolFromSmiles(smiles)
#     return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)

def fingerprints_to_tensor(fps):
    return torch.tensor([list(fp) for fp in fps], dtype=torch.float32)

def calculate_novelty(new_smiles_list):
    data = pd.read_csv("./data/sources/zinc250k/zinc250k_selfies.csv")
    known_smiles_list = data["smiles"].tolist()
    known_fps = fingerprints_to_tensor([smiles_to_fingerprint(smiles) for smiles in known_smiles_list])#.cuda()
    new_fps = fingerprints_to_tensor([smiles_to_fingerprint(smiles) for smiles in new_smiles_list])#.cuda()
    
    dot_product = torch.mm(new_fps, known_fps.t())
    norm_new = new_fps.sum(dim=1).unsqueeze(1)
    norm_known = known_fps.sum(dim=1).unsqueeze(0)
    similarity_matrix = dot_product / (norm_new + norm_known - dot_product)
    
    max_similarities, _ = similarity_matrix.max(dim=1)
    novelties = 1 - max_similarities
    
    return novelties.cpu().numpy()


def mol_prop(mol, prop):
    try:
        mol = Chem.MolFromSmiles(mol)
    except:
        return None
    # always remember to check if mol is None
    if mol is None:
        return None
    
    ## Basic Properties
    if prop == 'logP':
        return Descriptors.MolLogP(mol)
    elif prop == 'weight':
        return Descriptors.MolWt(mol)
    elif prop == 'qed':
        return Descriptors.qed(mol)
    elif prop == 'TPSA':
        return Descriptors.TPSA(mol)
    elif prop == 'HBA': # Hydrogen Bond Acceptor
        return Descriptors.NumHAcceptors(mol)
    elif prop == 'HBD': # Hydrogen Bond Donor
        return Descriptors.NumHDonors(mol)
    elif prop == 'rot_bonds': # rotatable bonds
        return Descriptors.NumRotatableBonds(mol)
    elif prop == 'ring_count':
        return Descriptors.RingCount(mol)
    elif prop == 'mr': # Molar Refractivity
        return Descriptors.MolMR(mol)
    elif prop == 'balabanJ':
        return Descriptors.BalabanJ(mol)
    elif prop == 'hall_kier_alpha':
        return Descriptors.HallKierAlpha(mol)
    elif prop == 'logD':
        return Descriptors.MolLogP(mol)
    elif prop == 'MR':
        return Descriptors.MolMR(mol)

    ## If Molecule is valid
    elif prop == 'validity':   
        # print(mol)
        return True
    
    ## Bond Counts
    elif prop == 'num_single_bonds':
        return sum([bond.GetBondType() == Chem.rdchem.BondType.SINGLE for bond in mol.GetBonds()])
    elif prop == 'num_double_bonds':
        return sum([bond.GetBondType() == Chem.rdchem.BondType.DOUBLE for bond in mol.GetBonds()])
    elif prop == 'num_triple_bonds':
        return sum([bond.GetBondType() == Chem.rdchem.BondType.TRIPLE for bond in mol.GetBonds()])
    elif prop == 'num_aromatic_bonds':
        return sum([bond.GetBondType() == Chem.rdchem.BondType.AROMATIC for bond in mol.GetBonds()])
    elif prop == 'num_rotatable_bonds': # rotatable bonds
        return Descriptors.NumRotatableBonds(mol)

    
    ## Common Atom Counts
    elif prop == 'num_carbon':
        return sum([atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()])
    elif prop == 'num_nitrogen':
        return sum([atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()])
    elif prop == 'num_oxygen':
        return sum([atom.GetAtomicNum() == 8 for atom in mol.GetAtoms()])
    elif prop == 'num_fluorine':
        return sum([atom.GetAtomicNum() == 9 for atom in mol.GetAtoms()])
    elif prop == 'num_phosphorus':
        return sum([atom.GetAtomicNum() == 15 for atom in mol.GetAtoms()])
    elif prop == 'num_sulfur':
        return sum([atom.GetAtomicNum() == 16 for atom in mol.GetAtoms()])
    elif prop == 'num_chlorine':
        return sum([atom.GetAtomicNum() == 17 for atom in mol.GetAtoms()])
    elif prop == 'num_bromine':
        return sum([atom.GetAtomicNum() == 35 for atom in mol.GetAtoms()])
    elif prop == 'num_iodine':
        return sum([atom.GetAtomicNum() == 53 for atom in mol.GetAtoms()])
    elif prop == "num_boron":
        return sum([atom.GetAtomicNum() == 5 for atom in mol.GetAtoms()])
    elif prop == "num_silicon":
        return sum([atom.GetAtomicNum() == 14 for atom in mol.GetAtoms()])
    elif prop == "num_selenium":
        return sum([atom.GetAtomicNum() == 34 for atom in mol.GetAtoms()])
    elif prop == "num_tellurium":
        return sum([atom.GetAtomicNum() == 52 for atom in mol.GetAtoms()])
    elif prop == "num_arsenic":
        return sum([atom.GetAtomicNum() == 33 for atom in mol.GetAtoms()])
    elif prop == "num_antimony":
        return sum([atom.GetAtomicNum() == 51 for atom in mol.GetAtoms()])
    elif prop == "num_bismuth":
        return sum([atom.GetAtomicNum() == 83 for atom in mol.GetAtoms()])
    elif prop == "num_polonium":
        return sum([atom.GetAtomicNum() == 84 for atom in mol.GetAtoms()])
    
    ## Functional groups
    elif prop == "num_benzene_ring":
        smarts = '[cR1]1[cR1][cR1][cR1][cR1][cR1]1'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_hydroxyl":
        smarts = '[OX2H]'   # Hydroxyl including phenol, alcohol, and carboxylic acid.
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_anhydride":
        smarts = '[CX3](=[OX1])[OX2][CX3](=[OX1])'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_aldehyde":
        smarts = '[CX3H1](=O)[#6]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_ketone":
        smarts = '[#6][CX3](=O)[#6]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_carboxyl":
        smarts = '[CX3](=O)[OX2H1]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_ester":
        smarts = '[#6][CX3](=O)[OX2H0][#6]'    # Ester Also hits anhydrides but won't hit formic anhydride.
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_amide":
        smarts = '[NX3][CX3](=[OX1])[#6]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_amine":
        smarts = '[NX3;H2,H1;!$(NC=O)]'    # Primary or secondary amine, not amide.
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_nitro":
        smarts = '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_halo":
        smarts = '[F,Cl,Br,I]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_thioether":
        smarts = '[SX2][CX4]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_nitrile":
        smarts = '[NX1]#[CX2]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_thiol":
        smarts = '[#16X2H]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_sulfide":
        smarts = '[#16X2H0]'    #  Won't hit thiols. Hits disulfides too.
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        exception = '[#16X2H0][#16X2H0]'
        matches_exception = mol.GetSubstructMatches(Chem.MolFromSmarts(exception))
        return len(matches) - len(matches_exception)
    elif prop == "num_disulfide":
        smarts = '[#16X2H0][#16X2H0]'    
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_sulfoxide":
        smarts = '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_sulfone":
        smarts = '[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_borane":
        smarts = '[BX3]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)

    else:
        raise ValueError(f'Property {prop} not supported')

def calculate_basic_property(smiles, prop):

    if prop == "heavy" or prop == "light":
        if mol_prop(smiles, 'weight') > 250:
            return "heavy" == prop
        else:
            return "light" == prop
        
    elif prop == "complex" or prop == "simple":
        if mol_prop(smiles, 'ring_count') > 3:
            if prop == "complex":
                return True
        if mol_prop(smiles, 'rot_bonds') > 3:
            if prop == "complex":
                return True
        if mol_prop(smiles, 'num_carbon') > 20:
            if prop == "complex":
                return True
            else:
                return False
        else:
            if prop == "complex":
                return False
            else:
                return True
    # below requires loading a simple model for prediction
    elif prop in ["toxic", "non-toxic", "toxicity", "non-toxicity"]:
        pass
    elif prop in ["high-boiling", "low-boiling", "high boiling point", "low bioling point"]:
        pass
    elif prop in ["high-melting", "low-melting", "high melting point", "low melting point"]:
        pass
    elif prop in ["water-soluble", "water-insoluble", "soluble in water", "insoluble in water"]:
        pass
    else:
        raise ValueError(f'Property {prop} not supported')
    
if __name__ == '__main__':
    smiles = 'CC#CC(C)C1CCCC1'

    print(mol_prop(smiles, 'num_single_bonds'))
