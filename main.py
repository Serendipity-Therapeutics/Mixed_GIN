from tqdm import tqdm
from tdc.benchmark_group import admet_group
from data.tdc_benchmark import ADMETBenchmarks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from data.data_loader import SMILESDataset
from utils.util import Scheduler, EarlyStopping
from models.gin_infomax import GINBinaryClassifier

get_admet_benchmarks = ADMETBenchmarks()
group = admet_group(path='data/')
selected_indices = [9]
for admet_benchmark in get_admet_benchmarks(selected_indices): 
    predictions_list= []
    for seed in tqdm([1,2,3,4,5]):
        benchmark = group.get(admet_benchmark)
        predictions = {}
        name = benchmark['name']
        train, test = benchmark['train_val'], benchmark['test']
        train, valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
        task, log_scale = get_admet_benchmarks(name)

        train_dataset = SMILESDataset(train)
        valid_dataset = SMILESDataset(valid)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GINBinaryClassifier()
        model.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = Scheduler(optimizer, factor=0.5, patience=3)
        early_stopping = EarlyStopping(patience=10, delta=0.001, path=f'checkpoint_{name}.pt')

        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            total = 0

            for smiles_batch, labels_batch in train_loader:
                smiles_batch = list(smiles_batch)
                labels_batch = labels_batch.float().unsqueeze(1).to(device)
                optimizer.zero_grad()
                outputs = model(smiles_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * labels_batch.size(0)
                preds = (outputs > 0.5).float()
                train_correct += (preds == labels_batch).sum().item()
                total += labels_batch.size(0)

            train_loss_epoch = train_loss / total
            train_acc_epoch = train_correct / total

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_labels = []
            all_preds = []

            with torch.no_grad():
                for smiles_batch, labels_batch in valid_loader:
                    smiles_batch = list(smiles_batch)
                    labels_batch = labels_batch.float().unsqueeze(1).to(device)
                    outputs = model(smiles_batch)
                    loss = criterion(outputs, labels_batch)

                    val_loss += loss.item() * labels_batch.size(0)
                    preds = (outputs > 0.5).float()
                    val_correct += (preds == labels_batch).sum().item()
                    val_total += labels_batch.size(0)

                    all_labels.extend(labels_batch.cpu().numpy())
                    all_preds.extend(outputs.cpu().numpy())

            val_loss_epoch = val_loss / val_total
            val_acc_epoch = val_correct / val_total
            auprc = average_precision_score(all_labels, all_preds)

            scheduler.step(val_loss_epoch)
            early_stopping(val_loss_epoch, model)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train loss: {train_loss_epoch:.4f}, acc: {train_acc_epoch:.4f} - "
                  f"Val loss: {val_loss_epoch:.4f}, acc: {val_acc_epoch:.4f}, AUPRC: {auprc:.4f}")

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        model.load_state_dict(torch.load(f'.ckpt/checkpoint_{name}_{seed}.pt'))

        test_dataset = SMILESDataset(test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        model.eval()
        y_pred_test = []
        with torch.no_grad():
            for smiles_batch, _ in test_loader:
                smiles_batch = list(smiles_batch)
                outputs = model(smiles_batch)
                y_pred_test.extend(outputs.cpu().numpy())
        y_pred_test = torch.tensor(y_pred_test)
            
        predictions[name] = y_pred_test
        predictions_list.append(predictions)

    results = group.evaluate_many(predictions_list)
    print('\n\n{}'.format(results))
    # {'caco2_wang': [6.328, 0.101]}