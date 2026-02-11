import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def test():
    print("hi")

# creates the graph for visualizing losses
def visualize_losses(losses): 
    losses_float = [float(loss.cpu().detach().numpy()) for loss in losses] 
    sns.lineplot(x=range(len(losses_float)), y=losses_float)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

# plots predicted over expected and prints evaluation metrics
def model_eval(model, test_loader, device):
    df = pd.DataFrame()
    model.eval()
    y_real_all = []
    y_pred_all = []

    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = test_batch.to(device)

            pred, embed = model(
                test_batch.x.float(),
                test_batch.edge_index,
                test_batch.batch
            )

            y_real_all.extend(test_batch.y.cpu().numpy().flatten())
            y_pred_all.extend(pred.cpu().numpy().flatten())
            
    df = pd.DataFrame({
        "y_real": y_real_all,
        "y_pred": y_pred_all
    })

    # plotting the predictions vs. observed variables
    plt = sns.scatterplot(data=df, x="y_real", y="y_pred")
    plt.set(xlim=(-7, 2))
    plt.set(ylim=(-7, 2))
    plt

    y_true = df["y_real"].values
    y_pred = df["y_pred"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, p_value = pearsonr(y_true, y_pred)
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")
    print(f"Pearson correlation: {pearson_corr:.3f} (p-value: {p_value:.3e})")

# pred. over observed for just 1 test batch
def model_eval_one(model, test_loader, device):
    test_batch = next(iter(test_loader))
    with torch.no_grad():
        test_batch.to(device)
        pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
        df = pd.DataFrame()
        df["y_real"] = test_batch.y.tolist()
        df["y_pred"] = pred.tolist()
    df["y_real"] = df["y_real"].apply(lambda row: row[0])
    df["y_pred"] = df["y_pred"].apply(lambda row: row[0])
    df

    # plotting the predictions vs. observed variables
    plt = sns.scatterplot(data=df, x="y_real", y="y_pred")
    plt.set(xlim=(-7, 2))
    plt.set(ylim=(-7, 2))
    plt