# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:15:49 2019

@author: hcji
"""

import os
import pickle
import numpy as np
from collections import defaultdict
from rdkit import Chem

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)
        
        
if __name__ == "__main__":
    
    import pandas as pd
    from DeepCCS.model.encoders import AdductToOneHotEncoder
    
    def preprocess(dataset, dir_input):
        
        train_smiles = list(dataset['SMILES'])
        train_adducts = dataset['Adducts']
        train_ccs = list(dataset['CCS'])
        
        adducts_encoder = AdductToOneHotEncoder()
        adducts_encoder.fit(train_adducts)
        adducts = adducts_encoder.transform(train_adducts)
    
        Smiles, molecules, adjacencies, properties = '', [], [], []
        for i, smi in enumerate(train_smiles):
            if '.' in smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            atoms = create_atoms(mol)
            i_jbond_dict = create_ijbonddict(mol)
        
            fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
            adjacency = create_adjacency(mol)
        
            Smiles += smi + '\n'
            molecules.append(fingerprints)
            adjacencies.append(adjacency)
            properties.append([[train_ccs[i]]])
    
        properties = np.array(properties)
        mean, std = np.mean(properties), np.std(properties)
        properties = np.array((properties - mean) / std)

    
        os.makedirs(dir_input, exist_ok=True)

        with open(dir_input + 'Smiles.txt', 'w') as f:
            f.write(Smiles)
        np.save(dir_input + 'molecules', molecules)
        np.save(dir_input + 'adducts', adducts)
        np.save(dir_input + 'adjacencies', adjacencies)
        np.save(dir_input + 'properties', properties)
        np.save(dir_input + 'mean', mean)
        np.save(dir_input + 'std', std)
        dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')
      
    radius = 1
    data = pd.read_csv('Data/data.csv')
    dir_input = ('Data/GNN_CCS/input/radius' + str(radius) + '/')
    
    preprocess(data, dir_input)
    print('The preprocess of the dataset has finished!')    