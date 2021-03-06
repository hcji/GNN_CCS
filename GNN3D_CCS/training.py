import timeit

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import GNN3D_CCS.preprocess as pp

# The setting of a neural network architecture.
dim=200
nadduct=4
layer_hidden=6
layer_output=3
layer_predict=3

# The setting for optimization.
batch_train=32
batch_test=32
lr=1e-3
lr_decay=0.99
decay_interval=10
iteration=30

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_atoms, dim, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_atom = nn.Embedding(N_atoms, dim)
        self.gamma = nn.ModuleList([nn.Embedding(N_atoms, 1)
                                    for _ in range(layer_hidden)])
        self.W_atom = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_predict = nn.ModuleList([nn.Linear(dim+nadduct, dim+nadduct)
                                       for _ in range(layer_predict)])
        self.W_property = nn.Linear(dim+nadduct, 1)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_atom[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def forward(self, inputs):

        """Cat or pad each input data for batch processing."""
        atoms, distance_matrices, molecular_sizes, adducts = inputs
        atoms = torch.cat(atoms)
        distance_matrix = self.pad(distance_matrices, 1e6)

        """GNN layer (update the atom vectors)."""
        atom_vectors = self.embed_atom(atoms)
        for l in range(layer_hidden):
            gammas = torch.squeeze(self.gamma[l](atoms))
            M = torch.exp(-gammas*distance_matrix**2)
            atom_vectors = self.update(M, atom_vectors, l)
            atom_vectors = F.normalize(atom_vectors, 2, 1)  # normalize.

        """Output layer."""
        for l in range(layer_output):
            atom_vectors = torch.relu(self.W_output[l](atom_vectors))

        """Molecular vector by sum of the atom vectors."""
        molecular_vectors = self.sum(atom_vectors, molecular_sizes)
        
        # combine with adduct information
        adducts = torch.stack(adducts)
        molecular_vectors = torch.cat((molecular_vectors, adducts),1)
        for l in range(layer_predict):
            molecular_vectors = torch.relu(self.W_predict[l](molecular_vectors))     

        """Molecular property."""
        properties = self.W_property(molecular_vectors)

        return properties

    def __call__(self, data_batch, train):

        inputs = data_batch[:-1]
        correct_properties = torch.cat(data_batch[-1])

        if train:
            predicted_properties = self.forward(inputs)
            loss = F.mse_loss(predicted_properties, correct_properties)
            return loss
        else:
            with torch.no_grad():
                predicted_properties = self.forward(inputs)
            ts = correct_properties.to('cpu').data.numpy()
            ys = predicted_properties.to('cpu').data.numpy()
            ts, ys = np.concatenate(ts), np.concatenate(ys)
            sum_absolute_error = sum(np.abs(ts-ys))
            return ts, ys, sum_absolute_error


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            loss = self.model(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        Ts, Ys, SAE = [], [], 0
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            ts, ys, sae = self.model(data_batch, train=False)
            SAE += sae
            Ts += ts.tolist()
            Ys += ys.tolist()
        MAE = SAE / N
        return np.array(Ts), np.array(Ys), MAE

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')
            
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def train_test_split(data_file, corrd_file, ratio=0.1):
    # data_file = 'Data/data.csv'
    # corrd_file = 'Data/GNN3D_CCS/3DCoord_CCS.txt'
    data = pd.read_csv(data_file)
    smiles = list(data['SMILES'])
    adducts = list(data['Adducts'])
    
    with open(corrd_file, 'r') as f:
        property_types = f.readline().strip().split()
        data_original = f.read().strip().split('\n\n')
    
    # include adducts
    for i,dt in enumerate(data_original):
        dt += ' ' + str(adducts[i])
        data_original[i] = dt
    
    smiles_unique = np.unique(smiles)
    
    n = int(ratio * len(smiles_unique))
    train_smiles, test_smiles = smiles_unique[n:], smiles_unique[:n]
    train_index = np.where([i in train_smiles for i in smiles])[0]
    test_index = np.where([i in test_smiles for i in smiles])[0]
    
    with open('Data/GNN3D_CCS/data_train.txt', 'w+') as train:
        train.write(str(property_types[0]) + ' Adducts')
        train.write('\n\n')
    with open('Data/GNN3D_CCS/data_test.txt', 'w+') as test:
        test.write(str(property_types[0]) + ' Adducts')
        test.write('\n\n')
    
    with open('Data/GNN3D_CCS/data_train.txt', 'a+') as train:
        for i in train_index:
            dt = data_original[i]
            train.write(dt)
            train.write('\n\n')
    with open('Data/GNN3D_CCS/data_test.txt', 'a+') as test:
        for i in test_index:
            dt = data_original[i]
            test.write(dt)
            test.write('\n\n')
    print('finished\n')
        

if __name__ == "__main__":
    '''
    (dataset, property, dim, layer_hidden, layer_output,
     batch_train, batch_test, lr, lr_decay, decay_interval, iteration,
     setting) = sys.argv[1:]
    (dim, layer_hidden, layer_output, batch_train, batch_test, decay_interval,
     iteration) = map(int, [dim, layer_hidden, layer_output, batch_train,
                            batch_test, decay_interval, iteration])
    
    '''
    
    train_test_split('Data/data.csv', 'Data/GNN3D_CCS/3DCoord_CCS.txt')
    
    dataset='GNN3D_CCS'
    file_model = 'Output/GNN3D_CCS/model.h5'
    # dataset=yourdataset

    # The molecular property to be learned.
    property='CCS'
    # property='HOMO(eV)'
    # property='LUMO(eV)'
    
    lr, lr_decay = map(float, [lr, lr_decay])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-'*100)

    print('Preprocessing the', dataset, 'dataset.')
    print('Just a moment......')
    (dataset_train, dataset_dev, dataset_test,
     N_atoms) = pp.create_datasets(dataset, property, device)
    print('-'*100)

    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-'*100)
    
    # load mean and std
    mean = np.load('Data/GNN3D_CCS/mean.npy')
    std = np.load('Data/GNN3D_CCS/std.npy')

    print('Creating a model.')
    torch.manual_seed(1234)  # initialize the model with a random seed.
    model = MolecularGraphNeuralNetwork(
            N_atoms, dim, layer_hidden, layer_output).to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    for i in range(layer_hidden):
        ones = nn.Parameter(torch.ones((N_atoms, 1))).to(device)
        model.gamma[i].weight.data = ones  # initialize gamma with ones.

    file_result = 'Output/GNN3D_CCS/result.txt'

    result = ('Epoch\tTime(sec)\tLoss_train(MSE)\t'
              'Error_dev(MAE)\tError_test(MAE)')
    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(1234)

    start = timeit.default_timer()

    for epoch in range(iteration):

        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        
        error_best = 99999
        
        loss_train = trainer.train(dataset_train)
        _, _, error_dev = tester.test(dataset_dev)
        _, _, error_test = tester.test(dataset_test)

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_train,
                                     error_dev, error_test]))
        tester.save_result(result, file_result)
        
        if error_dev <= error_best:
            error_best = error_dev
            tester.save_model(model, file_model)

        print(result)

    print('The training has finished!')
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_absolute_error
    
    true_ccs, pred_ccs, predictions_test = tester.test(dataset_test)
    true_ccs = true_ccs * std + mean
    pred_ccs = pred_ccs * std + mean

    r2 = r2_score(true_ccs, pred_ccs)
    mae = mean_absolute_error(true_ccs, pred_ccs)
    rmae = np.mean(np.abs(true_ccs - pred_ccs) / true_ccs) * 100    
