from collections import defaultdict
import os
import pickle
import sys
import timeit

import numpy as np

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor, Descriptors
from rdkit.Chem import MACCSkeys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# dictionary of atoms where a new element gets a new index
def create_atoms(mol, atom_dict):
    atoms = [atom_dict[a.GetSymbol()] for a in mol.GetAtoms()]
    return np.array(atoms)

# format from_atomIDx : [to_atomIDx, bondDict]
def create_ijbonddict(mol, bond_dict):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def create_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using WeisfeilerLehman-like algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        vertices = atoms
        for _ in range(radius):
            fingerprints = []
            for i, j_bond in i_jbond_dict.items():
                neighbors = [(vertices[j], bond) for j, bond in j_bond]
                fingerprint = (vertices[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            vertices = fingerprints

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency  = Chem.GetAdjacencyMatrix(mol)
    n          = adjacency.shape[0]

    adjacency  = adjacency + np.eye(n)
    degree     = sum(adjacency)
    d_half     = np.sqrt(np.diag(degree))
    d_half_inv = np.linalg.inv(d_half)
    adjacency  = np.matmul(d_half_inv,np.matmul(adjacency,d_half_inv))
    return np.array(adjacency)


def dump_dictionary(dictionary, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(dict(dictionary), f)


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_numpy(file_name):
    return np.load(file_name + '.npy', allow_pickle=True)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


dim            = 50
layer          = 2
batch          = 10
lr             = 1e-3
lr_decay       = 0.75
decay_interval = 20
iteration      = 100
extra_dim      = 20

(dim, layer, batch, decay_interval, iteration, extra_dim) = map(int, [dim, layer, batch, decay_interval, iteration, extra_dim])
lr, lr_decay = map(float, [lr, lr_decay])

class PathwayPredictor(nn.Module):

    def __init__(self, n_fingerprint):
        super(PathwayPredictor, self).__init__()
        self.embed_atom = nn.Embedding(n_fingerprint, dim)
        self.W_atom = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer)])
        self.W_property = nn.Linear(dim+extra_dim, 21)

    """Pad adjacency matrices for batch processing."""
    def pad(self, matrices, value):
        sizes = [d.shape[0] for d in matrices]
        D = sum(sizes)
        pad_matrices = value + np.zeros((D, D))
        m = 0
        for i, d in enumerate(matrices):
            s_i = sizes[i]
            pad_matrices[m:m+s_i, m:m+s_i] = d
            m += s_i
        return torch.FloatTensor(pad_matrices).to(device)

    def sum_axis(self, xs, axis):
        y = list(map(lambda x: torch.sum(x, 0), torch.split(xs, axis)))
        return torch.stack(y)

    def update(self, xs, adjacency, i):
        hs = torch.relu(self.W_atom[i](xs))
        return torch.matmul(adjacency, hs)

    def forward(self, inputs, sel_maccs):

        atoms, adjacency = inputs

        axis = list(map(lambda x: len(x), atoms))

        atoms = torch.cat(atoms)

        x_atoms = self.embed_atom(atoms)
        adjacency = self.pad(adjacency, 0)

        for i in range(layer):
            x_atoms = self.update(x_atoms, adjacency, i)

        extra_inputs = sel_maccs.to(device)
        y_molecules = self.sum_axis(x_atoms, axis)

        y_molecules = torch.cat((y_molecules,extra_inputs),1)
        z_properties = self.W_property(y_molecules)

        return z_properties

    def __call__(self, data_batch, train=True):

        sel_maccs = torch.FloatTensor(data_batch[-1])

        inputs, t_properties = data_batch[:-2], torch.cat(data_batch[-2])

        z_properties = self.forward(inputs, sel_maccs)

        if train:
            loss = F.binary_cross_entropy(torch.sigmoid(z_properties), t_properties)
            return loss
        else:
            zs = torch.sigmoid(z_properties).to('cpu').data.numpy()
            ts = t_properties.to('cpu').data.numpy()
            scores = list(map(lambda x: x, zs))
            labels = list(map(lambda x: (x>=0.5).astype(int), zs))
            return scores, labels, ts
        
class Trainer(object):

    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset_train):
        np.random.shuffle(dataset_train)
        N = len(dataset_train)
        loss_total = 0
        for i in range(0, N, batch):
            data_batch = list(zip(*dataset_train[i:i+batch]))
            loss = self.model(data_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total
    
    

class Tester(object):

    def __init__(self, model):
        self.model = model

    def test(self, dataset_test):

        N = len(dataset_test)
        score_list, label_list, t_list = [], [], []

        for i in range(0, N, batch):
            data_batch = list(zip(*dataset_test[i:i+batch]))
            scores, labels, ts = self.model(data_batch, train=False)
            score_list = np.append(score_list, scores)
            label_list = np.append(label_list, labels)
            t_list = np.append(t_list, ts)

        auc       = accuracy_score(t_list, label_list)
        precision = precision_score(t_list, label_list)
        recall    = recall_score(t_list, label_list)

        return auc, precision, recall
    
def create_file_with_smiles(file_prefix, dataset):
    #get the index of the labels where it is not 0 per row
    classes = [np.where(row != 0)[0].tolist() for row in dataset.y]
    # smiles to txt file with the index of the labels where it is not 0
    with open(f'{file_prefix}.txt', 'w') as f:
        for i, c in enumerate(classes):
            f.write(f"{dataset.smiles[i]}\t{','.join(map(str, c))}\n")

def generate_features(file_prefix, atom_dict=None, bond_dict=None, fingerprint_dict=None):
    radius = 2

    with open(f'{file_prefix}.txt', 'r') as f:
        data_list = f.read().strip().split('\n')

    """Exclude the data contains "." in the smiles, which correspond to non-bonds"""
    data_list = list(filter(lambda x: '.' not in x.strip().split()[0], data_list))
    N = len(data_list)

    print('Total number of molecules : %d' %(N))
    if atom_dict is None:
        atom_dict = defaultdict(lambda: len(atom_dict))
    if bond_dict is None:
        bond_dict = defaultdict(lambda: len(bond_dict))
    if fingerprint_dict is None:
        fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))

    Molecules, Adjacencies, Properties, MACCS_list = [], [], [], []

    max_MolMR, min_MolMR     = -1000, 1000
    max_MolLogP, min_MolLogP = -1000, 1000
    max_MolWt, min_MolWt     = -1000, 1000
    max_NumRotatableBonds, min_NumRotatableBonds = -1000, 1000
    max_NumAliphaticRings, min_NumAliphaticRings = -1000, 1000
    max_NumAromaticRings, min_NumAromaticRings   = -1000, 1000
    max_NumSaturatedRings, min_NumSaturatedRings = -1000, 1000

    for no, data in enumerate(data_list):

        print('/'.join(map(str, [no+1, N])))

        smiles, property_indices = data.strip().split('\t')
        property_s = property_indices.strip().split(',')

        property = np.zeros((1,21))
        for prop in property_s:
            property[0,int(prop)] = 1

        Properties.append(property)

        mol = Chem.MolFromSmiles(smiles)
        atoms = create_atoms(mol, atom_dict)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)

        fingerprints = create_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict)
        Molecules.append(fingerprints)

        adjacency = create_adjacency(mol)
        Adjacencies.append(adjacency)

        MACCS         = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))
        MACCS_ids     = np.zeros((20,))
        MACCS_ids[0]  = Descriptors.MolMR(mol)
        MACCS_ids[1]  = Descriptors.MolLogP(mol)
        MACCS_ids[2]  = Descriptors.MolWt(mol)
        MACCS_ids[3]  = Descriptors.NumRotatableBonds(mol)
        MACCS_ids[4]  = Descriptors.NumAliphaticRings(mol)
        MACCS_ids[5]  = MACCS[108]
        MACCS_ids[6]  = Descriptors.NumAromaticRings(mol)
        MACCS_ids[7]  = MACCS[98]
        MACCS_ids[8]  = Descriptors.NumSaturatedRings(mol)
        MACCS_ids[9]  = MACCS[137]
        MACCS_ids[10] = MACCS[136]
        MACCS_ids[11] = MACCS[145]
        MACCS_ids[12] = MACCS[116]
        MACCS_ids[13] = MACCS[141]
        MACCS_ids[14] = MACCS[89]
        MACCS_ids[15] = MACCS[50]
        MACCS_ids[16] = MACCS[160]
        MACCS_ids[17] = MACCS[121]
        MACCS_ids[18] = MACCS[149]
        MACCS_ids[19] = MACCS[161]

        if max_MolMR < MACCS_ids[0]:
            max_MolMR = MACCS_ids[0]
        if min_MolMR > MACCS_ids[0]:
            min_MolMR = MACCS_ids[0]

        if max_MolLogP < MACCS_ids[1]:
            max_MolLogP = MACCS_ids[1]
        if min_MolLogP > MACCS_ids[1]:
            min_MolLogP = MACCS_ids[1]

        if max_MolWt < MACCS_ids[2]:
            max_MolWt = MACCS_ids[2]
        if min_MolWt > MACCS_ids[2]:
            min_MolWt = MACCS_ids[2]

        if max_NumRotatableBonds < MACCS_ids[3]:
            max_NumRotatableBonds = MACCS_ids[3]
        if min_NumRotatableBonds > MACCS_ids[3]:
            min_NumRotatableBonds = MACCS_ids[3]

        if max_NumAliphaticRings < MACCS_ids[4]:
            max_NumAliphaticRings = MACCS_ids[4]
        if min_NumAliphaticRings > MACCS_ids[4]:
            min_NumAliphaticRings = MACCS_ids[4]

        if max_NumAromaticRings < MACCS_ids[6]:
            max_NumAromaticRings = MACCS_ids[6]
        if min_NumAromaticRings > MACCS_ids[6]:
            min_NumAromaticRings = MACCS_ids[6]

        if max_NumSaturatedRings < MACCS_ids[8]:
            max_NumSaturatedRings = MACCS_ids[8]
        if min_NumSaturatedRings > MACCS_ids[8]:
            min_NumSaturatedRings = MACCS_ids[8]

        MACCS_list.append(MACCS_ids)

    dir_input = (f'{file_prefix}_pathway/input'+str(radius)+'/')
    os.makedirs(dir_input, exist_ok=True)

    for n in range(N):
        for b in range(20):
            if b==0:
                MACCS_list[n][b] = (MACCS_list[n][b]-min_MolMR)/(max_MolMR-min_MolMR)
            elif b==1:
                MACCS_list[n][b] = (MACCS_list[n][b]-min_MolLogP)/(max_MolMR-min_MolLogP)
            elif b==2:
                MACCS_list[n][b] = (MACCS_list[n][b]-min_MolWt)/(max_MolMR-min_MolWt)
            elif b==3:
                MACCS_list[n][b] = (MACCS_list[n][b]-min_NumRotatableBonds)/(max_MolMR-min_NumRotatableBonds)
            elif b==4:
                MACCS_list[n][b] = (MACCS_list[n][b]-min_NumAliphaticRings)/(max_MolMR-min_NumAliphaticRings)
            elif b==6:
                MACCS_list[n][b] = (MACCS_list[n][b]-min_NumAromaticRings)/(max_MolMR-min_NumAromaticRings)
            elif b==8:
                MACCS_list[n][b] = (MACCS_list[n][b]-min_NumSaturatedRings)/(max_NumSaturatedRings-min_NumSaturatedRings)

    np.save(dir_input + 'molecules', np.array(Molecules, dtype=object))
    np.save(dir_input + 'adjacencies', np.array(Adjacencies, dtype=object))
    np.save(dir_input + 'properties', np.array(Properties, dtype=object))
    np.save(dir_input + 'maccs', np.asarray(MACCS_list))

    dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')

    print('The preprocess has finished!')

    return atom_dict, bond_dict, fingerprint_dict

def load_features(file_prefix, radius):
    dir_input = (f'{file_prefix}_pathway/input'+str(radius)+'/')

    molecules    = load_tensor(dir_input + 'molecules', torch.LongTensor)
    adjacencies  = load_numpy(dir_input + 'adjacencies')

    import numpy as np
    properties = np.load(dir_input + 'properties.npy', allow_pickle=True)  # Load as object
    properties = np.array([np.asarray(x, dtype=np.float32) for x in properties])  # Convert to float
    t_properties = torch.FloatTensor(properties)  # Now compatible
    t_properties = t_properties.to(device)
    maccs        = load_numpy(dir_input + 'maccs')

    # with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
    #     fingerprint_dict = pickle.load(f)

    dataset = list(zip(molecules, adjacencies, t_properties, maccs))

    return dataset
    
def benchmark(datasets):

    folds = []
    reps  = []

    f1_scores = []
    precision_scores = []
    recall_scores = []
    weighted_f1_scores = []
    weighted_precision_scores = []
    weighted_recall_scores = []
    
    rep = 0
    for tuple_datasets in datasets:
        fold_idx = 0
        for train_dataset, validation_dataset, test_dataset_ in tuple_datasets:
            train_valid_merged = train_dataset.merge([validation_dataset])
            create_file_with_smiles('train', train_valid_merged)
            create_file_with_smiles('test', test_dataset_)
            atom_dict, bond_dict, fingerprint_dict = generate_features('train')
            unknown          = 150
            n_fingerprint    = len(fingerprint_dict) + unknown
            generate_features('test', atom_dict, bond_dict, fingerprint_dict)

            train_dataset = load_features('train', 2)
            test_dataset = load_features('test', 2)

            torch.manual_seed(1234)

            model   = PathwayPredictor(n_fingerprint).to(device)
            trainer = Trainer(model)
            tester  = Tester(model)

            dir_output = ('pathway/output/')
            os.makedirs(dir_output, exist_ok=True)

            print('Training...')
            print('Epoch \t Time(sec) \t Loss_train \t AUC_dev \t AUC_test \t Precision \t Recall')

            start = timeit.default_timer()

            for epoch in range(iteration):
                if (epoch+1) % decay_interval == 0:
                    trainer.optimizer.param_groups[0]['lr'] *= lr_decay

                loss    = trainer.train(train_dataset)

                lr_rate = trainer.optimizer.param_groups[0]['lr']

                end  = timeit.default_timer()
                time = end - start

                print('%d \t %.4f \t %.4f' %(epoch, time, loss))

            data_batch = list(zip(*test_dataset[:]))

            sel_maccs            = torch.FloatTensor(data_batch[-1])
            inputs, t_properties = data_batch[:-2], torch.cat(data_batch[-2])
            z_properties         = model.forward(inputs, sel_maccs)

            p_properties = torch.sigmoid(z_properties)

            p_properties = p_properties.data.to('cpu').numpy()
            t_properties = t_properties.data.to('cpu').numpy()

            p_properties[p_properties<0.5]  = 0
            p_properties[p_properties>=0.5] = 1

            y_true = t_properties
            y_pred = p_properties

            train_labels = np.array(train_valid_merged.y)
            valid_labels = np.array(test_dataset_.y)

            # Check which labels have at least one '1' in both train and validation datasets
            labels_with_ones = []
            for i in range(train_labels.shape[1]):  # Assuming labels are in one-hot encoded format
                if np.any(train_labels[:, i] == 1) and np.any(valid_labels[:, i] == 1):
                    labels_with_ones.append(i)

            y_pred = y_pred[:, labels_with_ones]
            y_true = y_true[:, labels_with_ones]

            f1_score_ = f1_score(y_pred=y_pred, y_true=y_true, average='macro')
            recall_score_ = recall_score(y_pred=y_pred, y_true=y_true, average='macro')
            precision_score_ = precision_score(y_pred=y_pred, y_true=y_true, average='macro')

            wf1_score = f1_score(y_pred=y_pred, y_true=y_true, average='weighted')
            wrecall_score = recall_score(y_pred=y_pred, y_true=y_true, average='weighted')
            wprecision_score = precision_score(y_pred=y_pred, y_true=y_true, average='weighted')

            # export to a dataframe and to a csv file
            import pandas as pd

            f1_scores.append(f1_score_)
            precision_scores.append(precision_score_)
            recall_scores.append(recall_score_)
            weighted_f1_scores.append(wf1_score)
            weighted_precision_scores.append(wprecision_score)
            weighted_recall_scores.append(wrecall_score)


            folds.append(fold_idx)
            reps.append(rep)

            fold_idx += 1
            # create dataframe with the results
            results = {
                'fold': folds,
                'f1_score': f1_scores,
                'precision_score': precision_scores,
                'recall_score': recall_scores,
                'wF1': weighted_f1_scores,
                'wPrecision': weighted_precision_scores,
                'wRecall': weighted_recall_scores,
                'rep': reps
            }
            df = pd.DataFrame(results)
            # save the dataframe to a csv file
            df.to_csv(f'results.csv', index=False)
        rep += 1

import pickle
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

with open("splits.pkl", "rb") as f:
    splits = pickle.load(f)

benchmark(splits)


