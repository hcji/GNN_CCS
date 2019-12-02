# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:34:56 2019

@author: hcji
"""

import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
rdescriptor_path = '''source('{}')'''.format('GNN3D_CCS/R/ChemmineOB.R')
robjects.r(rdescriptor_path)
convertFormat = robjects.globalenv['OB_convertFormat']


def generate_MOP_input(mol, mop_path, key='BONDS'):
    mol = Chem.AddHs(mol)
    mol_block = Chem.MolToMolBlock(mol)
    Chem.MolToMolFile(mol, mop_path.replace('.mop', '.mol'))
    mop = str(convertFormat("MOL", "MOP", mol_block)[0])
    mop = mop.replace('PUT KEYWORDS HERE', key)
    with open(mop_path, 'w') as file:
        print(mop, file=file)
    print ('done')


def call_MOPAC(mop_path, mopac_path):
    cmd = str(mopac_path)
    cmd += ' '
    cmd += os.path.join(os.getcwd(), mop_path)
    subprocess.call(cmd)
    print ('done')


def parser_MOPAC_bond_order(mopac_out):
    # mopac_out = 'example/balsalazide.out'
    with open(mopac_out) as t:
        res = t.readlines()
    start = 10**6
    for i, txt in enumerate(res):
        if 'FINAL HEAT OF FORMATION' in txt:
            hh = txt.split('=')[2]
            hh = hh.split(' ')
            for h in hh:
                try:
                    heat = float(h)
                except:
                    pass
        if 'BOND ORDERS' in txt:
            start = i + 1
        if '*****' in txt and i > start:
            end = i
            break
            
    atom1 = []
    atom2 = []
    order = []
    for i in range(start, end):
        txt = res[i]
        if '(' not in txt:
            continue
        else:
            n = []
            txt += res[i+1]
            txt = txt.split(' ')
            for j in txt:
                try:
                    n.append(float(j))
                except:
                    pass
            for j, b in enumerate(n):
                if j == 0:
                    f = int(b)
                elif j % 2 == 1:
                    atom1.append(f)
                    atom2.append(int(b))
                elif j % 2 == 0:
                    order.append(float(b))
    bond_order = pd.DataFrame({'atom1': atom1, 'atom2': atom2, 'bond_order': order})

    print ('done')
    return {'form_heat': heat, 'bond_order': bond_order}
    

def parser_MOPAC_coordinates(mopac_out):
    # mopac_out = 'mopac_working/unknow_mol.out'
    with open(mopac_out) as t:
        res = t.readlines()
    start = 10**6
    for i, txt in enumerate(res):
        if 'CARTESIAN COORDINATES' in txt:
            start = i + 1
        if txt == '\n' and i > start + 4:
            end = i
            break
    output = pd.DataFrame(columns=['index', 'element', 'x', 'y', 'z'])
    for i in range(start+3, end):
        txt = res[i]
        txt = txt.replace('\n', '')
        txt = txt.split(' ')
        txt = [x for x in txt if x != '']
        output.loc[len(output)] = txt
    return output
        

def run_MOPAC(mol, mopac_path, aim='coordinates', mol_name=None):
    os.mkdir('mopac_working')
    working_dir = os.path.join(os.getcwd(), 'mopac_working')
    if mol_name is None:
        mol_name = 'unknow_mol'
    mop_path = os.path.join(working_dir, mol_name + '.mop')
    print ('generating input file ...')
    generate_MOP_input(mol, mop_path, key='BONDS')
    print ('calling mopac software ...')
    call_MOPAC(mop_path, mopac_path)
    print ('generating result ...')
    mopac_out = mop_path.replace('.mop', '.out')
    if aim == 'coordinates':
        output = parser_MOPAC_coordinates(mopac_out)
    else:
        output = parser_MOPAC_bond_order(mopac_out)
    shutil.rmtree(working_dir)
    return output


def draw_mol(mol):
    dr = rdMolDraw2D.MolDraw2DSVG(800,800)
    dr.SetFontSize(0.3)
    op = dr.drawOptions()
    for i in range(mol.GetNumAtoms()) :
        op.atomLabels[i] = mol.GetAtomWithIdx(i).GetSymbol() + str((i+1))
    AllChem.Compute2DCoords(mol)
    dr.DrawMolecule(mol)
    dr.FinishDrawing()
    svg = dr.GetDrawingText()
    SVG(svg)


if __name__ == '__main__':
    
    from tqdm import tqdm
    
    with open('Data/GNN3D_CCS/coordinates.txt', 'w+') as txt:
        txt.write('CCS')
        txt.write('\n')
        txt.write('\n')
    
    mopac_path = 'E:/MOPAC2016/MOPAC2016_with_a_window_for_WINDOWS_64_bit/MOPAC2016.exe'
    data = pd.read_csv('Data/data.csv')
    for i in tqdm(range(len(data))):
        name = str(i)
        smi = data['SMILES'][i]
        mol = Chem.MolFromSmiles(smi)
        
        res = run_MOPAC(mol, mopac_path, mol_name=name)
        with open('Data/GNN3D_CCS/coordinates.txt', 'a+') as txt:
            txt.write(name)
            txt.write('\n')
            for j in range(len(res)):
                txt.write(res['element'][j] + ' ' + res['x'][j] + ' ' + res['y'][j] + ' ' + res['z'][j])
                txt.write('\n')
            txt.write(str(data['CCS'][i]))
            txt.write('\n')
            txt.write('\n')