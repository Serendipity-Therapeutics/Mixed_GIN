# trainer.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, mean_absolute_error

from tdc.benchmark_group import admet_group
from data.tdc_benchmark import ADMETBenchmarks
from data.data_loader import SMILESDataset
from Aizen_project.Mixed_GIN.models.gin_model import GINBinaryClassifier, GINRegressor
from utils.util import Scheduler, EarlyStopping
from utils.loss_fn import HuberLoss

from wandb_logger import WandbLogger  

PROJECT_NAME = "ADMET_GIN_Project"  

def train_and_evaluate(
    model_kind="gin_supervised_infomax",
    benchmark_key="cyp2c9_veith",
    seeds=[1, 2, 3, 4, 5],
    epochs=100,
    batch_size=32,
    learning_rate=1e-4
):

    run_name = f"{model_kind}_{benchmark_key}"
    config = {
        "model_kind": model_kind,
        "benchmark": benchmark_key,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seeds": seeds
    }
    logger = WandbLogger(
        project=PROJECT_NAME,
        run_name=run_name,
        config=config
    )

    get_admet_benchmarks = ADMETBenchmarks()
    group = admet_group(path="data/")

    benchmark = group.get(benchmark_key)
    name = benchmark["name"]
    task, log_scale = get_admet_benchmarks(name)

    predictions_list = []
    os.makedirs(".ckpt", exist_ok=True)

    for seed in tqdm(seeds, desc=f"[{benchmark_key} | {model_kind}] Seeds"):
        train_df, test_df = benchmark["train_val"], benchmark["test"]
        train_df, valid_df = group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed
        )

        if task == "regression" and log_scale:
            train_df["Y"] = train_df["Y"].apply(lambda x: np.log(x + 1))
            valid_df["Y"] = valid_df["Y"].apply(lambda x: np.log(x + 1))
            test_df["Y"]  = test_df["Y"].apply(lambda x: np.log(x + 1))

        train_dataset = SMILESDataset(train_df)
        valid_dataset = SMILESDataset(valid_df)
        test_dataset  = SMILESDataset(test_df)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if task == "binary":
            model = GINBinaryClassifier(transformer_kind=model_kind)
            criterion = nn.BCELoss()
        else:
            model = GINRegressor(transformer_kind=model_kind)
            criterion = HuberLoss(delta=1.0, reduction='mean')

        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        scheduler = Scheduler(optimizer, factor=0.5, patience=3)
        ckpt_path = f".ckpt/checkpoint_{name}_{model_kind}_{seed}.pt"
        early_stopper = EarlyStopping(patience=10, delta=0.001, path=ckpt_path)

        for epoch in range(epochs):
            # TRAIN
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for smiles_batch, labels_batch in train_loader:
                smiles_batch = list(smiles_batch)
                labels_batch = labels_batch.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                outputs = model(smiles_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                bs = labels_batch.size(0)
                train_loss += loss.item() * bs
                train_total += bs

                if task == "binary":
                    preds = (outputs > 0.5).float()
                    train_correct += (preds == labels_batch).sum().item()

            train_loss_epoch = train_loss / train_total
            if task == "binary":
                train_acc_epoch = train_correct / train_total
            else:
                train_acc_epoch = None

            # -------- VALID --------
            model.eval()
            val_loss = 0.0
            val_total = 0
            val_correct = 0
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for smiles_batch, labels_batch in valid_loader:
                    smiles_batch = list(smiles_batch)
                    labels_batch = labels_batch.float().unsqueeze(1).to(device)

                    outputs = model(smiles_batch)
                    loss = criterion(outputs, labels_batch)

                    val_loss += loss.item() * labels_batch.size(0)
                    val_total += labels_batch.size(0)

                    label_cpu = labels_batch.cpu().numpy().flatten()
                    pred_cpu  = outputs.cpu().numpy().flatten()
                    all_labels.extend(label_cpu)
                    all_preds.extend(pred_cpu)

                    if task == "binary":
                        preds = (outputs > 0.5).float()
                        val_correct += (preds == labels_batch).sum().item()

            val_loss_epoch = val_loss / val_total

            if task == "binary":
                val_acc_epoch = val_correct / val_total
                val_auprc = average_precision_score(all_labels, all_preds)
                val_mae = None
            else:
                val_acc_epoch = None
                val_auprc = None

                if log_scale:
                    exp_labels = np.exp(all_labels) - 1
                    exp_preds  = np.exp(all_preds)  - 1
                    val_mae = mean_absolute_error(exp_labels, exp_preds)
                else:
                    val_mae = mean_absolute_error(all_labels, all_preds)

            if task == "binary":
                print(f"[Epoch {epoch+1}/{epochs} | seed={seed}] "
                      f"train_loss={train_loss_epoch:.4f}, val_loss={val_loss_epoch:.4f}, "
                      f"train_acc={train_acc_epoch:.4f}, val_acc={val_acc_epoch:.4f}, "
                      f"val_auprc={val_auprc:.4f}")
            else:
                print(f"[Epoch {epoch+1}/{epochs} | seed={seed}] "
                      f"train_loss={train_loss_epoch:.4f}, val_loss={val_loss_epoch:.4f}, "
                      f"val_mae={val_mae:.4f}")

            log_dict = {
                "epoch": epoch + 1,
                "seed": seed,
                "train_loss": train_loss_epoch,
                "val_loss": val_loss_epoch,
            }
            if task == "binary":
                log_dict["train_acc"] = train_acc_epoch
                log_dict["val_acc"]   = val_acc_epoch
                log_dict["val_auprc"] = val_auprc
            else:
                log_dict["val_mae"] = val_mae

            logger.log(log_dict)

            scheduler.step(val_loss_epoch)
            early_stopper(val_loss_epoch, model)
            if early_stopper.early_stop:
                print(f"Early stopping (epoch={epoch+1}) at seed={seed}, "
                      f"benchmark={benchmark_key}, model={model_kind}")
                break

        # TEST EVALUATION
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        test_preds = []
        with torch.no_grad():
            for smiles_batch, _ in test_loader:
                smiles_batch = list(smiles_batch)
                outputs = model(smiles_batch)
                test_preds.extend(outputs.cpu().numpy().flatten())

        test_preds = np.array(test_preds)
        predictions = {}
        if task == "binary":
            predictions[name] = torch.tensor(test_preds)
        else:
            if log_scale:
                predictions[name] = torch.tensor(np.exp(test_preds) - 1)
            else:
                predictions[name] = torch.tensor(test_preds)

        predictions_list.append(predictions)

    # TDC evaluate
    results = group.evaluate_many(predictions_list)
    final_score = results.get(name, None)

    if final_score:
        score_mean, score_std = final_score
        if task == "binary":
            logger.log({
                "test_auprc_mean": score_mean,
                "test_auprc_std": score_std
            })
            print(f"[TEST] {benchmark_key} => AUPRC mean={score_mean:.4f}, std={score_std:.4f}")
        else:
            logger.log({
                "test_mae_mean": score_mean,
                "test_mae_std": score_std
            })
            print(f"[TEST] {benchmark_key} => MAE mean={score_mean:.4f}, std={score_std:.4f}")

    logger.finish()


if __name__ == "__main__":

    transformer_kinds = [
        "gin_supervised_infomax",
        "gin_supervised_contextpred",
        "gin_supervised_edgepred",
        "gin_supervised_masking",
    ]
    get_admet_benchmarks = ADMETBenchmarks()
    all_benchmarks = get_admet_benchmarks(None) 
    seeds = [1, 2, 3, 4, 5]

    for model_kind in transformer_kinds:
        for benchmark_key in all_benchmarks:
            train_and_evaluate(
                model_kind=model_kind,
                benchmark_key=benchmark_key,
                seeds=seeds,
                epochs=200,
                batch_size=32,
                learning_rate=1e-4
            )

    print("All runs done!")