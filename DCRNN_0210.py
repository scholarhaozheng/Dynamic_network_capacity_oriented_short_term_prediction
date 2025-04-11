from torch_geometric_temporal import StaticGraphTemporalSignal
import pickle
import os
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.nn.recurrent import GConvGRU


from tqdm import tqdm

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, hidden_units, output_dim, K):
        super(RecurrentGCN, self).__init__()
        # self.recurrent = DCRNN(node_features, hidden_units, K)
        self.recurrent = GConvGRU(node_features, hidden_units, K)
        self.linear = torch.nn.Linear(hidden_units, output_dim)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.selu(h)
        h = self.linear(h)
        return h

def run_PYGT(base_dir, RGCN_node_features, RGCN_hidden_units, RGCN_output_dim, RGCN_K, lr, epoch_num, train_ratio):
    train_dir_path = os.path.join(base_dir, f'train_signal_dict_OD.pkl')
    with open(train_dir_path, 'rb') as f:
        signal_dict = pickle.load(f, errors='ignore')

    train_dataset = StaticGraphTemporalSignal(
        features=signal_dict["features"],
        targets=signal_dict["targets"],
        additional_feature1=signal_dict["additional_feature"],
        edge_index=signal_dict["edge_index"],
        edge_weight=signal_dict["edge_weight"]
    )

    #from torch_geometric_temporal.signal import temporal_signal_split
    ##train_dataset, test_dataset = temporal_signal_split(signal, train_ratio=train_ratio)

    test_dir_path = os.path.join(base_dir, f'test_signal_dict_OD.pkl')
    with open(test_dir_path, 'rb') as f:
        signal_dict = pickle.load(f, errors='ignore')

    test_dataset = StaticGraphTemporalSignal(
        features=signal_dict["features"],
        targets=signal_dict["targets"],
        additional_feature1=signal_dict["additional_feature"],
        edge_index=signal_dict["edge_index"],
        edge_weight=signal_dict["edge_weight"]
    )

    with open(os.path.join(base_dir, 'test_dataset_OD.pkl'), 'wb') as f:
        pickle.dump(test_dataset, f)

    RecurrentGCN_model = RecurrentGCN(node_features=RGCN_node_features, hidden_units=RGCN_hidden_units,
                                      output_dim=RGCN_output_dim, K=RGCN_K)

    optimizer = torch.optim.Adam(RecurrentGCN_model.parameters(), lr=lr)

    RecurrentGCN_model.train()
    for epoch in tqdm(range(epoch_num)):
        cost_PINN = 0
        total_mape = 0
        for snap_time, snapshot in enumerate(train_dataset):
            y_hat = RecurrentGCN_model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost_PINN = cost_PINN + torch.mean((y_hat - snapshot.y) ** 2)
            non_zero_mask = snapshot.y != 0
            mape = torch.mean(
                torch.abs((snapshot.y[non_zero_mask] - y_hat[non_zero_mask]) / snapshot.y[non_zero_mask])) * 100
            total_mape += mape

        cost_PINN = cost_PINN / (snap_time + 1)
        cost_PINN.backward()
        optimizer.step()
        optimizer.zero_grad()

        avg_mape = total_mape / (snap_time + 1)

        print(f"Epoch {epoch + 1}/{epoch_num}, MSE: {cost_PINN.item():.4f}, MAPE: {avg_mape.item():.4f}%")

    model_save_path = os.path.join(base_dir, f'RecurrentGCN_model_OD.pth')
    torch.save(RecurrentGCN_model.state_dict(), model_save_path)

    hyperparameters = {
        "RGCN_node_features": RGCN_node_features,
        "RGCN_hidden_units": RGCN_hidden_units,
        "RGCN_output_dim": RGCN_output_dim,
        "RGCN_K": RGCN_K
    }

    hyperparams_save_path = os.path.join(base_dir, f'hyperparameters_OD.pkl')
    with open(hyperparams_save_path, 'wb') as f:
        pickle.dump(hyperparameters, f)


def test_PYGT(base_dir):
    test_dir_path = os.path.join(base_dir, f'test_dataset_OD.pkl')
    with open(test_dir_path, 'rb') as f:
        test_dataset = pickle.load(f, errors='ignore')

    hyperparams_path = os.path.join(base_dir, f'hyperparameters_OD.pkl')
    with open(hyperparams_path, 'rb') as f:
        hyperparameters = pickle.load(f)

    RGCN_node_features = hyperparameters['RGCN_node_features']
    RGCN_hidden_units = hyperparameters['RGCN_hidden_units']
    RGCN_output_dim = hyperparameters['RGCN_output_dim']
    RGCN_K = hyperparameters['RGCN_K']

    class RecurrentGCN(torch.nn.Module):
        def __init__(self, node_features, hidden_units, output_dim, K):
            super(RecurrentGCN, self).__init__()
            # self.recurrent = DCRNN(node_features, hidden_units, K)
            self.recurrent = GConvGRU(node_features, hidden_units, K)
            self.linear = torch.nn.Linear(hidden_units, output_dim)

        def forward(self, x, edge_index, edge_weight):
            h = self.recurrent(x, edge_index, edge_weight)
            h = F.selu(h)
            h = self.linear(h)
            return h

    RecurrentGCN_model = RecurrentGCN(node_features=RGCN_node_features, hidden_units=RGCN_hidden_units, output_dim=RGCN_output_dim,
                                      K=RGCN_K)

    model_path = os.path.join(base_dir, f"RecurrentGCN_model_OD.pth")
    RecurrentGCN_model.load_state_dict(torch.load(model_path))

    RecurrentGCN_model.eval()
    cost_PINN = 0
    with torch.no_grad():
        mae, rmse, mape_net = 0.0, 0.0, 0.0
        for snap_time, snapshot in enumerate(test_dataset):
            y_hat = RecurrentGCN_model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)

            # Remove zero entries for evaluation
            non_zero_mask = snapshot.y != 0

            # MAE calculation
            mae += torch.mean(torch.abs(snapshot.y[non_zero_mask] - y_hat[non_zero_mask]))

            # RMSE calculation
            rmse += torch.sqrt(torch.mean((snapshot.y[non_zero_mask] - y_hat[non_zero_mask]) ** 2))

            # MAPE_net calculation
            mape_net += torch.mean(
                torch.abs((snapshot.y[non_zero_mask] - y_hat[non_zero_mask]) / snapshot.y[non_zero_mask])) * 100

        # Averaging the metrics across all snapshots
        mae = mae / (snap_time + 1)
        rmse = rmse / (snap_time + 1)
        mape_net = mape_net / (snap_time + 1)

        mae = mae.item()
        rmse = rmse.item()
        mape_net = mape_net.item()

    print("TEST MAE: {:.4f}".format(mae))
    print("TEST RMSE: {:.4f}".format(rmse))
    print("TEST MAPE_net: {:.4f}".format(mape_net))


city_place="suzhou"
base_dir = f"data{os.path.sep}{city_place}"
if city_place=="suzhou":
    RGCN_node_features = 154
    RGCN_output_dim = 154
else:
    RGCN_node_features = 26
    RGCN_output_dim = 26

RGCN_hidden_units = 6
RGCN_K = 2
lr = 0.01
epoch_num = 200
train_ratio = 0.8
run_PYGT(base_dir, RGCN_node_features, RGCN_hidden_units, RGCN_output_dim, RGCN_K, lr, epoch_num, train_ratio)
test_PYGT(base_dir)
