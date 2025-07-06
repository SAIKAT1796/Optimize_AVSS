import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import itertools
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

# --------------------- AVSS Model ---------------------
class AVSSModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(AVSSModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.dropout(lstm_out[:, -1, :])
        return self.fc(x)

# --------------------- Data Loader ---------------------
def load_avss_data(samples=500, input_size=10, output_size=2, validation_split=0.2):
    X = torch.rand(samples, 5, input_size)
    y = torch.randint(0, output_size, (samples,))
    dataset = TensorDataset(X, y)
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

# --------------------- Training & Evaluation ---------------------
def train_avss_model(model, optimizer, criterion, train_loader, epochs, device="cpu"):
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    return loss.item()

def evaluate_avss_model(model, val_loader, criterion, device="cpu"):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            loss = criterion(model(batch_X), batch_y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# --------------------- Taguchi Method ---------------------
def run_taguchi_experiment(L9_array, param_ranges, model_class, data_loader_func, num_workers=0):
    results = []
    for i, params in enumerate(L9_array):
        hyperparams = dict(zip(param_ranges.keys(), params))
        print(f"\nExperiment {i + 1}: {hyperparams}")

        train_dataset, val_dataset = data_loader_func()
        train_loader = DataLoader(train_dataset, batch_size=int(hyperparams["batch_size"]), shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=int(hyperparams["batch_size"]), shuffle=False, num_workers=num_workers)

        model = model_class(10, int(hyperparams["lstm_hidden_size"]), 2, hyperparams["dropout"])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])

        train_loss = train_avss_model(model, optimizer, criterion, train_loader, int(hyperparams["epochs"]))
        val_loss = evaluate_avss_model(model, val_loader, criterion)

        results.append({**hyperparams, "train_loss": train_loss, "val_loss": val_loss})

    return results

def analyze_taguchi_results(results, param_ranges):
    results_array = np.array([[result[param] for param in param_ranges.keys()] + [result["val_loss"]] for result in results], dtype=object)
    sn_ratios = {}
    for i, param in enumerate(param_ranges.keys()):
        values = np.unique(results_array[:, i])
        sn_ratios[param] = [-10 * np.log10(np.mean([float(row[-1]) for row in results_array if row[i] == val])) for val in values]

    plt.figure(figsize=(12, 6))
    for i, param in enumerate(sn_ratios):
        plt.subplot(2, 3, i+1)
        plt.plot(param_ranges[param], sn_ratios[param], marker='o')
        plt.title(f"S/N vs. {param}")
        plt.xlabel(param)
        plt.ylabel("S/N Ratio")
    plt.tight_layout()
    plt.show()

    anova_results = {}
    for i, param in enumerate(param_ranges.keys()):
        values = np.unique(results_array[:, i])
        groups = [[float(row[-1]) for row in results_array if row[i] == val] for val in values]
        f_stat, p_val = f_oneway(*groups)
        anova_results[param] = (f_stat, p_val)

    return sn_ratios, anova_results, results_array

# --------------------- Bayesian Optimization ---------------------
def bayesian_optimization(init_points, n_iter, param_ranges, data_loader_func, model_class, initial_points, num_workers=0):
    def objective(batch_size, learning_rate, dropout, lstm_hidden_size, weight_decay, epochs):
        hyperparams = {
            "batch_size": int(round(batch_size)),
            "learning_rate": learning_rate,
            "dropout": dropout,
            "lstm_hidden_size": int(round(lstm_hidden_size)),
            "weight_decay": weight_decay,
            "epochs": int(round(epochs))
        }

        train_dataset, val_dataset = data_loader_func()
        train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"], shuffle=False, num_workers=num_workers)

        model = model_class(10, hyperparams["lstm_hidden_size"], 2, hyperparams["dropout"])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])

        train_avss_model(model, optimizer, criterion, train_loader, hyperparams["epochs"])
        val_loss = evaluate_avss_model(model, val_loader, criterion)

        return -val_loss

    pbounds = {
        "batch_size": (min(param_ranges["batch_size"]), max(param_ranges["batch_size"])),
        "learning_rate": (min(param_ranges["learning_rate"]), max(param_ranges["learning_rate"])),
        "dropout": (min(param_ranges["dropout"]), max(param_ranges["dropout"])),
        "lstm_hidden_size": (min(param_ranges["lstm_hidden_size"]), max(param_ranges["lstm_hidden_size"])),
        "weight_decay": (min(param_ranges["weight_decay"]), max(param_ranges["weight_decay"])),
        "epochs": (min(param_ranges["epochs"]), max(param_ranges["epochs"]))
    }

    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)

    for row in initial_points:
        params = dict(zip(pbounds.keys(), row[:-1]))
        optimizer.probe(params=params, lazy=True)

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    print("\nBayesian Optimization Results:")
    for i, res in enumerate(optimizer.res):
        print(f"Iteration {i}: Target = {-res['target']:.4f}, Params: {res['params']}")

    best_params = optimizer.max["params"]
    best_loss = -optimizer.max["target"]

    return best_params, best_loss

# --------------------- Main ---------------------
if __name__ == '__main__':
    PARAM_RANGES = {
        "batch_size": [16, 32, 64],
        "learning_rate": [1e-5, 1e-4, 1e-3],
        "dropout": [0.1, 0.3, 0.5],
        "lstm_hidden_size": [32, 64, 128],
        "weight_decay": [1e-6, 1e-5, 1e-4],
        "epochs": [25, 40, 50]
    }

    L9_orthogonal_array = np.array([
        [16, 1e-5, 0.1, 32, 1e-6, 25],
        [16, 1e-4, 0.3, 64, 1e-5, 40],
        [16, 1e-3, 0.5, 128, 1e-4, 50],
        [32, 1e-5, 0.3, 128, 1e-4, 40],
        [32, 1e-4, 0.5, 32, 1e-6, 50],
        [32, 1e-3, 0.1, 64, 1e-5, 25],
        [64, 1e-5, 0.5, 64, 1e-4, 50],
        [64, 1e-4, 0.1, 128, 1e-6, 25],
        [64, 1e-3, 0.3, 32, 1e-5, 40]
    ], dtype=object)

    taguchi_results = run_taguchi_experiment(L9_orthogonal_array, PARAM_RANGES, AVSSModel, load_avss_data, num_workers=2)
    sn_ratios, anova_results, results_array = analyze_taguchi_results(taguchi_results, PARAM_RANGES)

    print("\nBest Hyperparameters from Taguchi:")
    best_result_taguchi = min(taguchi_results, key=lambda x: x["val_loss"])
    print(best_result_taguchi)

    print("\nRunning Bayesian Optimization...")
    best_params, best_loss = bayesian_optimization(
        init_points=5,
        n_iter=15,
        param_ranges=PARAM_RANGES,
        data_loader_func=load_avss_data,
        model_class=AVSSModel,
        initial_points=results_array,
        num_workers=2
    )

    print("\nBest Parameters from Bayesian Optimization:", best_params)
    print("Best Validation Loss:", best_loss)