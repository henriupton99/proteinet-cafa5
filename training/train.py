import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import gc
import torch
from preprocessing.dataset import ProteinSequenceDataset
from training.models import LinearModel

from sklearn.model_selection import KFold
from torchmetrics.classification import MultilabelF1Score

embeds_dim = {
    "T5" : 1024,
    "ProtBERT" : 1024,
    "ESM2" : 2560
}

def train_model(
    aspect : str,
    embeddings_source : str,
    num_classes : int,
    hidden_dim : int,
    k_folds : int,
    batch_size : int,
    n_epochs : int,
    learning_rate : float,
    device : str,
    validation_mode : bool
    ):
    """Function to train an expert model (BPO, CCO, or MFO) based on a given input aspect name.
    NB : You need to first train with validation_mode=True in order to validate your model on several folds
    Then run again your model with validation_mode=False to train on all data for the final inference model

    Args:
        aspect (str): name of the aspect to train a model
        embeddings_source (str): source of the embeddings to consider ("T5","ProtBERT","EMS2")
        num_classes (int): number of classes for the model
        hidden_dim (int) : dimension of the hidden layer for the model
        k_folds (int) : number of folds in the Kfold Cross Validation routine
        batch_size (int) : batch size of the dataloader 
        n_epochs (int): number of epochs for training
        learning_rate (float): learning rate for optimizer
        device (str): device for training (cuda for GPU, or cpu for CPU)
        validation_mode (bool): option for validation of final training of the model
    """
    
    train_dataset = ProteinSequenceDataset(aspect = aspect, datatype="train", embeddings_source = embeddings_source)  
    models_path = "./models/"    
    aspects_path = "./models/{}/".format(aspect)
    history_path = "./models/history/"
    for path in [models_path, aspects_path, history_path]:
        if not os.path.exists(path):
            os.mkdir(path)
        
    losses_history = pd.DataFrame({"epoch" : [e for e in range(1,n_epochs+1)]})
    scores_history = pd.DataFrame({"epoch" : [e for e in range(1,n_epochs+1)]})
    
    if validation_mode == True:
        kfold = KFold(n_splits=k_folds, shuffle=True) 
        for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):

            print("="*25)
            print("FOLD {}".format(fold+1))
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_subsampler
                )
            val_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=val_subsampler
                )
            gc.collect()

            model = LinearModel(input_dim=embeds_dim[embeddings_source],hidden_dim=hidden_dim, num_classes=num_classes).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

            df_weights = pd.read_csv(
                "./data/train-labels-targets/weights_{}.csv".format(aspect)
            )
            classes_weights = torch.tensor(df_weights.weight.values, dtype=torch.float64)

            CrossEntropy = torch.nn.CrossEntropyLoss(weight=classes_weights).to(device)
            f1_score = MultilabelF1Score(num_labels=num_classes).to(device)

            print("BEGIN TRAINING...")
            train_loss_history=[]
            val_loss_history=[]

            train_f1score_history=[]
            val_f1score_history=[]

            for epoch in range(n_epochs):
                print("EPOCH ", epoch+1)
                ## TRAIN PHASE :
                losses = []
                scores = []
                model.train()
                for embed, targets in tqdm(train_dataloader):
                    embed, targets = embed.to(device), targets.to(device)
                    optimizer.zero_grad()
                    preds = model(embed)
                    loss= CrossEntropy(preds, targets)
                    score=f1_score(preds, targets)
                    losses.append(loss.item()) 
                    scores.append(score.item())
                    loss.backward()
                    optimizer.step()
                avg_loss = np.mean(losses)
                avg_score = np.mean(scores)
                print("Running Average TRAIN Loss : ", avg_loss)
                print("Running Average TRAIN F1-Score : ", avg_score)
                train_loss_history.append(avg_loss)
                train_f1score_history.append(avg_score)

                ## VALIDATION PHASE : 
                model.eval()
                losses = []
                scores = []
                for embed, targets in val_dataloader:
                    embed, targets = embed.to(device), targets.to(device)
                    preds = model(embed)
                    loss= CrossEntropy(preds, targets)
                    score=f1_score(preds, targets)
                    losses.append(loss.item())
                    scores.append(score.item())
                avg_loss = np.mean(losses)
                avg_score = np.mean(scores)
                print("Running Average VAL Loss : ", avg_loss)
                print("Running Average VAL F1-Score : ", avg_score)
                val_loss_history.append(avg_loss)
                val_f1score_history.append(avg_score)

                #scheduler.step(avg_loss)
                print("\n")

            print("TRAINING FINISHED")
            print("FINAL TRAINING SCORE FOLD {}: ".format(fold+1), train_f1score_history[-1])
            print("FINAL VALIDATION SCORE : {}".format(fold+1), val_f1score_history[-1])

            losses_history["train_fold_{}".format(fold+1)] = train_loss_history
            losses_history["val_fold_{}".format(fold+1)] = val_loss_history

            scores_history["train_fold_{}".format(fold+1)] = train_f1score_history
            scores_history["val_fold_{}".format(fold+1)] = val_f1score_history

            torch.save(model.state_dict(), "./models/{}/expert_model_fold_{}.pt".format(aspect,fold+1))
            print("MODEL SAVED AT ./models/{}/expert_model_fold_{}.pt".format(aspect,fold+1))

            del model, train_dataloader, val_dataloader
            gc.collect()
            print("="*25)

        losses_history.to_csv(history_path + "losses_history_{}.csv".format(aspect),index=None)
        scores_history.to_csv(history_path + "scores_history_{}.csv".format(aspect),index=None)
        del losses_history, scores_history
        del train_dataset
        gc.collect()
    
    if validation_mode == False:
        train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True
                )
        gc.collect()

        model = LinearModel(input_dim=embeds_dim[embeddings_source],hidden_dim=hidden_dim, num_classes=num_classes).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

        df_weights = pd.read_csv(
            "./data/train-labels-targets/weights_{}.csv".format(aspect)
        )
        classes_weights = torch.tensor(df_weights.weight.values, dtype=torch.float64)

        CrossEntropy = torch.nn.CrossEntropyLoss(weight=classes_weights).to(device)
        f1_score = MultilabelF1Score(num_labels=num_classes).to(device)

        print("BEGIN TRAINING...")
        train_loss_history=[]

        train_f1score_history=[]

        for epoch in range(n_epochs):
            print("EPOCH ", epoch+1)
            ## TRAIN PHASE :
            losses = []
            scores = []
            model.train()
            for embed, targets in tqdm(train_dataloader):
                embed, targets = embed.to(device), targets.to(device)
                optimizer.zero_grad()
                preds = model(embed)
                loss= CrossEntropy(preds, targets)
                score = f1_score(preds, targets)
                losses.append(loss.item()) 
                scores.append(score.item())
                loss.backward()
                optimizer.step()
            avg_loss = np.mean(losses)
            avg_score = np.mean(scores)
            print("Running Average TRAIN Loss : ", avg_loss)
            print("Running Average TRAIN F1-Score : ", avg_score)
            train_loss_history.append(avg_loss)
            train_f1score_history.append(avg_score)
        print("TRAINING FINISHED")
        print("FINAL TRAINING SCORE : ", train_f1score_history[-1])

        losses_history["train"] = train_loss_history
        scores_history["train"] = train_f1score_history

        torch.save(model.state_dict(), "./models/{}/expert_model.pt".format(aspect))
        print("MODEL SAVED AT ./models/{}/expert_model.pt".format(aspect))

        del model, train_dataloader
        gc.collect()
        print("="*25)

        losses_history.to_csv(history_path + "losses_history_{}.csv".format(aspect),index=None)
        scores_history.to_csv(history_path + "scores_history_{}.csv".format(aspect),index=None)
        del losses_history, scores_history
        del train_dataset
        gc.collect()
    
    print("TRAINING FINISHED ! :D")
    return