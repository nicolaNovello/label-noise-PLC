import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
import torch.nn.functional as F
import torch
from torch import nn
import csv
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import random
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
import scipy.stats
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class ToTensor(object):
    def __call__(self, sample):
        x, y = sample
        return (torch.tensor([np.array(x)]).float(),
                torch.tensor([y]))


class MyDataset(Dataset):
    def __init__(self, x_dataset, y_dataset, transform=None):
        self.transform = transform
        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.data = []
        for i in range(len(x_dataset)):
            self.data.append((x_dataset.iloc[i], y_dataset.iloc[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class NoisyModulation(Dataset):
    def __init__(self, is_train, x, y, noise_type, r, num_classes, transform):
        self.x = x
        self.y = y
        self.noise_type = noise_type
        self.r = r
        self.num_classes = num_classes
        self.transform = transform

        if is_train:
            t = self.load_noise_label()
            self.y = t.tolist()

    def load_transition_matrix(self):

        if self.noise_type == "symm":
            C = 1 - self.r
            N = self.r / (self.num_classes - 1)
            if self.num_classes == 2:
                return torch.tensor([[C, N],
                                    [N, C]])
            elif self.num_classes == 4:
                return torch.tensor([[C, N, N, N],
                                     [N, C, N, N],
                                     [N, N, C, N],
                                     [N, N, N, C]])
        elif self.noise_type == "sparse":
            C = 1 - self.r
            N = self.r
            if self.num_classes == 2:
                return torch.tensor([[C, N],
                                     [N, C]])
            elif self.num_classes == 4:
                return torch.tensor([[C, N, 0, 0],
                                     [N, C, 0, 0],
                                     [0, 0, C, N],
                                     [0, 0, N, C]])
        elif self.noise_type == "unif":
            s = np.random.uniform(0, self.r, self.num_classes).tolist()
            if self.num_classes == 2:
                return torch.tensor([[1-s[1], s[1]],
                                     [s[0], 1-s[0]]])
            elif self.num_classes == 4:
                uniform = [s.copy(), s.copy(), s.copy(), s.copy()]
                for i in range(self.num_classes):
                    tmp_red_s = s.copy()
                    tmp_red_s.pop(i)
                    uniform[i][i] = 1 - np.sum(tmp_red_s)
                uniform = torch.tensor(uniform).type(torch.float32)
                return uniform

    def load_noise_label(self):
        self.y = torch.tensor(self.y, dtype=torch.long)
        y_onehot = nn.functional.one_hot(self.y, self.num_classes).type(torch.float32)
        transition_matrix = self.load_transition_matrix()
        y_noisy = torch.matmul(y_onehot, transition_matrix).squeeze()
        samples = torch.multinomial(y_noisy, num_samples=1)
        return samples

    def __getitem__(self, index):
        sample = (self.x[index], self.y[index])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim, vectorize_input=False):
        super(Discriminator, self).__init__()

        self.vectorize_input = vectorize_input

        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
        )

    def forward(self, input_tensor):
        if self.vectorize_input:
            input_tensor = input_tensor.reshape(-1, input_tensor.shape[2]**2)
        output_tensor = self.main(input_tensor)
        return output_tensor


class CombinedArchitecture(nn.Module):
    def __init__(self, single_architecture, cost_function_v=1):
        super(CombinedArchitecture, self).__init__()
        self.div_to_act_func = {
            2: nn.Sigmoid(),
            3: nn.Identity(),
            5: nn.Sigmoid(),
        }
        self.cost_function_version = cost_function_v
        self.single_architecture = single_architecture
        self.final_activation = self.div_to_act_func[cost_function_v]

    def forward(self, input_tensor_1, input_tensor_2):
        intermediate_1 = self.single_architecture(input_tensor_1)
        output_tensor_1 = self.final_activation(intermediate_1)
        intermediate_2 = self.single_architecture(input_tensor_2)
        output_tensor_2 = self.final_activation(intermediate_2)

        return output_tensor_1, output_tensor_2


def load_matlab_file(path, verbose=True):
    mat = scipy.io.loadmat(path)
    if verbose:
        print("len(mat): ", len(mat))
        print("mat.keys: ", mat.keys())
    return mat


def preprocess_matlab_data(dataset, M=2, SNR=0.1, test_size=1000, compute_ml=False, noisy=True, noise_type="symm", r=0.1):
    L = 1
    n_rows = 1
    n_columns = dataset["ChQ09"].shape[1]
    a1 = dataset["ChQ09"][0, :]
    b1 = dataset["ChQ09"][1, :]
    c1 = dataset["ChQ09"][2, :]
    d1 = dataset["ChQ09"][3, :]


    if M == 4:
        Ztx = [0, 50, 200, 1e9]
    elif M == 2:
        Ztx = [0, 1e9]
    else:
        print("Not implemented for M != 2 and M != 4.")

    p = test_size
    Z = [np.divide(np.multiply(a1, Z_tmp) + b1, np.multiply(c1, Z_tmp) + d1) for Z_tmp in Ztx]

    # Shunt impedance
    Zs = 50
    Vg = np.sqrt(50*(10**(-50/10)))
    Vn = np.sqrt(50 * pow(10, (-50 - SNR)/10))

    # Shunt voltages
    Vs = [Vg * Zs / (Z_tmp + Zs * np.ones((Z_tmp.shape))) for Z_tmp in Z]

    Vs_n = [np.tile(Vs_tmp, (p,1)) + Vn*np.random.normal(size=(p*n_rows, n_columns)) + \
            1j*Vn*np.random.normal(size=(p*n_rows, n_columns)) for Vs_tmp in Vs]

    Vs_real = np.real(Vs_n)
    Vs_imag = np.imag(Vs_n)

    ## Construction of the dataset
    X_total_Ms = [np.concatenate((Vs_real[i], Vs_imag[i]), axis=1) for i in range(M)]
    Y_total_Ms = [np.ones((p*n_rows,1), dtype=np.int32)*i for i in range(M)]
    X_total = np.concatenate(X_total_Ms, axis=0)
    Y_total = np.concatenate(Y_total_Ms, axis=0)
    if compute_ml:
        y_predict = np.zeros(M*L*p)
        VM = [np.concatenate([np.real(VsX), np.imag(VsX)], axis=0) for VsX in Vs]
        template_M = [np.divide(Vi, np.linalg.norm(Vi)) for Vi in VM]
        for i in range(M * L * p):
            r_M = [np.dot(X_total[i,:], template_i.T) for template_i in template_M]
            y_predict[i] = np.argmax(r_M)

        y_predict = np.reshape(y_predict, Y_total.shape)
        e = y_predict - Y_total
        ber_ml = 1 - len(np.where(e==0)[0])/len(e)
    else:
        ber_ml = 0  # Not used in this case, use the same as before

    X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size=0.1, random_state=0, stratify=Y_total)

    print("X_train.shape:{}, X_test.shape:{} ".format(X_train.shape, X_test.shape))
    print("Y_train.shape:{}, Y_test.shape:{}".format(Y_train.shape, Y_test.shape))

    ## Normalize datasets
    scaler = StandardScaler()
    sc_x = scaler.fit(X_total)
    X_train = sc_x.transform(X_train)
    X_test = sc_x.transform(X_test)
    n_features = X_train.shape[1]
    composed_transform = transforms.Compose([ToTensor()])

    if noisy:
        train_dataset = NoisyModulation(is_train=True, x=X_train, y=Y_train, noise_type=noise_type, r=r, num_classes=M, transform=composed_transform)
        test_dataset = NoisyModulation(is_train=False, x=X_test, y=Y_test, noise_type=noise_type, r=r, num_classes=M, transform=composed_transform)
    else:
        train_dataset = MyDataset(from_numpy_to_dataframe(X_train),
                                  from_numpy_to_dataframe(Y_train),
                                  composed_transform)
        test_dataset = MyDataset(from_numpy_to_dataframe(X_test),
                                 from_numpy_to_dataframe(Y_test),
                                 composed_transform)

    return train_dataset, test_dataset, n_features, ber_ml


def impedance_modulation_SNR(main_proc_params, main_opt_params, channels_all, cost_function_v, test_size,
                             compute_ml=False, results_dict={}, noise_type="symm", r="0.1"):

    model_name = "impedance_modulation" + "_cost_" + str(cost_function_v)

    ber_vec = []
    BER_maxL_vec = []
    H_x_vec = []
    H_x_y_vec = []
    MI_vec = []
    P_error_vec = []
    for SNR in main_proc_params['SNR_vec']:
        print("SNR: ", SNR)
        train_dataset, test_dataset, n_features, ber_ml = preprocess_matlab_data(channels_all, main_proc_params['M'],
                                                                                SNR=SNR, test_size=test_size,
                                                                                compute_ml=compute_ml, noisy=main_proc_params['noisy'], noise_type=noise_type, r=r)
        print("FINISHED PRE-PROCESSING DATA...")
        input_dim = n_features
        output_dim = main_proc_params['M']
        batch_size_imp = 64
        epochs_imp = 10
        model = Discriminator(input_dim=input_dim, output_dim=output_dim)
        combined = CombinedArchitecture(model, cost_function_v=cost_function_v)
        combined = combined.to(main_proc_params['device'])
        trained_model = train_model(model_name, combined, train_dataset, cost_function_v=cost_function_v,
                                    num_classes=output_dim,
                                    batch_size=batch_size_imp, epochs=epochs_imp, device=main_proc_params['device'],
                                    verbose=False, save_epochs=[epochs_imp],
                                    save_training_loss=False, lr=main_opt_params['learning_rate'],
                                    alpha=main_proc_params['alpha'], random_seed=main_proc_params["random_seed"])
        accuracy, _, H_x, H_x_y, MI, P_error = test_model(trained_model, test_dataset, cost_function_v=cost_function_v, device=main_proc_params['device'], MI_estimate=False)
        ber_vec.append(1 - accuracy)
        BER_maxL_vec.append(ber_ml)
        H_x_vec.append(H_x)
        H_x_y_vec.append(H_x_y)
        MI_vec.append(MI)
        P_error_vec.append(P_error)

    results_dict['BER_maxL'] = BER_maxL_vec
    results_dict['SNR_vec'] = main_proc_params['SNR_vec']
    results_dict['ber_all_SNR'] = ber_vec
    results_dict['MI_all_SNR'] = MI_vec
    results_dict['H_x_all_SNR'] = H_x_vec
    results_dict['H_x_y_all_SNR'] = H_x_y_vec
    results_dict['P_error_all_SNR'] = P_error_vec

    return results_dict


def train_model(model_name, model, train_dataset, cost_function_v=1, num_classes=10, batch_size=100, epochs=10, device="cpu",
                verbose=True, save_epochs=[], save_training_loss=False, lr=0.001, alpha=1, random_seed=0):

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    model.train()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    losses = []
    for epoch in range(epochs):
        if verbose:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Starting epoch training at time =", current_time)
            print("EPOCH: ", epoch+1)
        loss_batch = []
        total = 0
        correct = 0
        for sample_batched in train_dataloader:
            data_rx = sample_batched[0].to(device).squeeze()
            data_tx = sample_batched[1].to(device).long().squeeze()
            current_batch_size = len(sample_batched[0])
            optimizer.zero_grad()
            data_y = get_random_batch(train_dataset, batch_size=current_batch_size).to(device)
            out_1, out_2 = model(data_rx, data_y)
            loss = compute_loss_divergence(cost_function_v, out_1, out_2, data_tx, num_classes, current_batch_size, alpha, device)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.item())
            R_all = obtain_posterior_from_net_out(out_1, cost_function_v)
            _, predicted = R_all.max(1)
            correct += predicted.eq(data_tx).sum().item()
            total += data_tx.size(0)
        print("Epoch: {}; Loss: {}; Accuracy: {}".format(epoch+1, np.mean(loss_batch), 100 * correct/total))
        losses.append(np.mean(loss_batch))

        if epoch in save_epochs:
            torch.save(model.state_dict(), "NetModels/{}_epoch_{}_costfunc_{}_seed_{}.pth".format(model_name, epoch, cost_function_v, random_seed))
            print("Saving network {}_epoch_{}_costfunc_{}_seed_{}.pth".format(model_name, epoch, cost_function_v, random_seed))
    if save_training_loss:
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss cost function v{}".format(cost_function_v))
        plt.savefig("LossPlots/Loss cost function v{}_epochs{}.png".format(cost_function_v, epochs))
    return model


def test_model(model, test_dataset, cost_function_v=1, device="cpu", MI_estimate=False, num_classes=4):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    model.eval()
    test_size = len(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)

    with torch.no_grad():
        for sample_batched in test_dataloader:
            data_rx = sample_batched[0].to(device).squeeze()
            data_tx = sample_batched[1].to(device).squeeze()
            D_all, _ = model(data_rx, data_rx)  # get all density-ratios
            R_all = obtain_posterior_from_net_out(D_all, cost_function_v)
            _, predicted = R_all.squeeze().max(1)
            R_all = R_all.squeeze().cpu().detach().numpy()
            map_result = R_all.argmax(axis=1)
            n_correct_pred = len(np.where(np.equal(map_result, data_tx.squeeze().cpu().detach().numpy()))[0])
            accuracy = n_correct_pred / test_size
            print("TEST accuracy: ", accuracy)
            if MI_estimate:
                BER_maxL = np.zeros((1, test_size))
                h_x_y = np.zeros((test_size, 1))
                L1_norm = np.expand_dims(np.sum(R_all, axis=1), axis=-1) * np.ones((1, np.size(R_all, axis=1)))
                R_all = R_all / L1_norm
                latent_dim = 1
                alphabet = range(num_classes)
                training_samples = from_digit_to_zero_mean_bits(alphabet, int(np.log2(num_classes)))
                for i in range(data_rx.shape[0]):
                    data_rx = data_rx.detach().numpy()
                    max_idx_genie = get_max_idx_loglikelihood(np.expand_dims(data_rx[i, :], axis=0), training_samples)
                    logical_bits_genie = training_samples[max_idx_genie, :] == data_tx[i,:].detach().numpy()
                    BER_maxL[0, i] = 1 - sum(logical_bits_genie) / latent_dim
                    D_value_1, _ = model(torch.Tensor(data_rx[i, :]), torch.Tensor(data_rx[i, :]))
                    R = obtain_posterior_from_net_out(D_value_1, cost_function_v)
                    R = R.detach().numpy()
                    L1_single_norm = np.expand_dims(np.sum(R, axis=1), axis=-1) * np.ones((1, np.size(R, axis=1)))
                    R = R / L1_single_norm
                    h_x_y[i] = -R[0, :].dot(np.log2(R[0, :].T))
                BER_maxL = np.sum(BER_maxL) / (test_size * latent_dim)
                P_x = np.mean(R_all, axis=0)
                H_x = -P_x.dot(np.log2(P_x))
                H_x_y = np.nanmean(h_x_y, axis=0)[0]
                MI = H_x - H_x_y
                P_error = 1 - np.mean(np.max(R_all, axis=1), axis=0)
                return accuracy, BER_maxL, H_x, H_x_y, MI, P_error
            else:
                return accuracy, 0, 0, 0, 0, 0


def compute_loss_divergence(cost_function_v, out_1, out_2, data_tx, num_classes, current_batch_size, alpha, device):
    loss_fn_3 = nn.CrossEntropyLoss()

    data_tx_categorical = torch.Tensor(to_categorical(data_tx, t_tensor=True, num_classes=num_classes))

    if cost_function_v == 2:  # gan
        loss = gan_cost_fcn(out_1, out_2, data_tx_categorical, num_classes, device=device)
    elif cost_function_v == 3:  # cross-entropy
        loss = loss_fn_3(out_1.squeeze(), data_tx.squeeze().long())
    elif cost_function_v == 5:  # sl
        loss = sl_cost_fcn(out_1, out_2, data_tx_categorical, num_classes, alpha)

    return loss


def from_numpy_to_dataframe(numpy_dataset):
    df = pd.DataFrame(numpy_dataset).reset_index(drop=True)
    return df


class MyException(Exception):
    pass


def get_random_batch(dataset, batch_size=32):
    train_dataloader_random = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    my_testiter = iter(train_dataloader_random)
    random_batch, target = next(my_testiter)
    return random_batch


def obtain_posterior_from_net_out(D, cost_function_v):
    if cost_function_v == 2 or cost_function_v == 5:
        R = (1-D)/D
    elif cost_function_v == 3:
        R = torch.exp(D)  # because linear layer is used, which can be negative. For expressing prob. it needs to be positive
    return R


def from_digit_to_zero_mean_bits(x,k):
    M = len(x)
    output = np.zeros((M,k))
    for i in x:
        output[i,:] = np.transpose(np.fromstring(np.binary_repr(i, width=k), np.int8) - 48)
    output = 2*output-1
    return output


def get_max_idx_loglikelihood(y,x):
    N = np.size(x,0)
    distances = np.zeros((N,1))
    for i in range(N):
        distances[i] = np.linalg.norm(y[0,:]-x[i,:])
    return np.argmin(distances)


def to_categorical(y, num_classes, t_tensor=False, dtype="uint8"):
    if t_tensor:
        return F.one_hot(y, num_classes=num_classes)
    else:
        return np.eye(num_classes, dtype=dtype)[y.astype(int).squeeze()]


def gan_cost_fcn(out_1, out_2, digits, num_classes, device="cpu", t_tensor=True):
    loss_fn = nn.BCELoss()
    loss_fn_2 = nn.BCELoss(reduction='none')
    batch_size = out_1.shape[0]
    valid = np.ones((batch_size, num_classes))
    non_valid = np.zeros((batch_size, num_classes))
    loss_1 = loss_fn_2(out_1.squeeze(), torch.Tensor(non_valid).to(device))
    loss_1 = torch.matmul(loss_1, torch.transpose(digits.float(), 0, 1).to(device))
    loss_1 = torch.diagonal(loss_1, 0)
    loss_1 = torch.mean(loss_1)
    loss_2 = loss_fn(out_2.squeeze(), torch.Tensor(valid).to(device))
    loss = loss_1 + loss_2
    return loss


def sl_first(y_pred, data_tx, num_classes, t_tensor=True):
    loss_1 = torch.matmul(y_pred, torch.transpose(data_tx.float(), 0, 1))
    loss_1 = torch.diagonal(loss_1, 0)
    loss_1 = torch.mean(loss_1)
    return loss_1


def sl_second(y_pred):
    log_pred = torch.log(y_pred) - y_pred
    sum_log_pred = torch.mean(log_pred, dim=1)
    loss = torch.mean(sum_log_pred)
    return -loss


def sl_cost_fcn(out_1, out_2, data_tx, num_classes, alpha):
    loss_1 = sl_first(out_1.squeeze(), data_tx, num_classes)
    loss_2 = sl_second(out_2.squeeze())
    loss = loss_1 + alpha * loss_2
    return loss


def save_data_and_figures(saving_path, cost_fcn, results_dict, mode, noisy=True, noise_type="symm", r=0.1):
    if noisy:
        tmp_path_saving_data = os.path.join(saving_path, "cf_{}_mode_{}_noise_{}_r_{}.csv".format(cost_fcn, mode, noise_type, r))
    else:
        tmp_path_saving_data = os.path.join(saving_path, "cf_{}_mode_{}.csv".format(cost_fcn, mode))
    save_dict_lists_csv(tmp_path_saving_data, results_dict)


def save_dict_lists_csv(path, dictionary):
    with open(path, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dictionary.keys())
        writer.writerows(zip(*dictionary.values()))