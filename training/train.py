import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torchmetrics.classification import MultilabelF1Score
#from torchmetrics.classification import MultilabelAccuracy

from preprocessing.dataset import ProteinSequenceDataset
from training.models import LinearModel, CNN1D

embeds_dim = {
    "T5" : 1024,
    "ProtBERT" : 1024,
    "ESM2" : 2560
}

def train_model(
    aspect,
    embeddings_source,
    model_type,
    num_classes,
    train_size,
    batch_size,
    n_epochs,
    lr,
    device
    ):
    
    train_dataset = ProteinSequenceDataset(aspect = aspect, datatype="train", embeddings_source = embeddings_source)
    
    train_set, val_set = random_split(train_dataset, lengths = [int(len(train_dataset)*train_size), len(train_dataset)-int(len(train_dataset)*train_size)])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    if model_type == "linear":
        model = LinearModel(input_dim=embeds_dim[embeddings_source], num_classes=num_classes).to(device)
    if model_type == "convolutional":
        model = CNN1D(input_dim=embeds_dim[embeddings_source], num_classes=num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1, threshold=0.001, verbose=True)
    CrossEntropy = torch.nn.CrossEntropyLoss()
    #f1_score = MultilabelF1Score(num_labels=num_classes).to(device)

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
        for embed, targets in tqdm(train_dataloader):
            embed, targets = embed.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(embed)
            loss= CrossEntropy(preds, targets)
            #score=f1_score(preds, targets)
            losses.append(loss.item()) 
            #scores.append(score.item())
            loss.backward()
            optimizer.step()
        avg_loss = np.mean(losses)
        avg_score = np.mean(scores)
        print("Running Average TRAIN Loss : ", avg_loss)
        print("Running Average TRAIN F1-Score : ", avg_score)
        train_loss_history.append(avg_loss)
        train_f1score_history.append(avg_score)
        
        ## VALIDATION PHASE : 
        losses = []
        scores = []
        for embed, targets in val_dataloader:
            embed, targets = embed.to(device), targets.to(device)
            preds = model(embed)
            loss= CrossEntropy(preds, targets)
            #score=f1_score(preds, targets)
            losses.append(loss.item())
            #scores.append(score.item())
        avg_loss = np.mean(losses)
        avg_score = np.mean(scores)
        print("Running Average VAL Loss : ", avg_loss)
        print("Running Average VAL F1-Score : ", avg_score)
        val_loss_history.append(avg_loss)
        val_f1score_history.append(avg_score)
        
        #scheduler.step(avg_loss)
        print("\n")
        
    print("TRAINING FINISHED")
    print("FINAL TRAINING SCORE : ", train_f1score_history[-1])
    print("FINAL VALIDATION SCORE : ", val_f1score_history[-1])
        
    losses_history = pd.DataFrame(data={"train" : train_loss_history, "val" : val_loss_history})
    scores_history = pd.DataFrame(data={"train" : train_f1score_history, "val" : val_f1score_history})
    
    torch.save(model.state_dict(), "./models/expert_model_"+aspect+".pt")
    losses_history.to_csv("./postprocessing/train_history/losses/losses_history_"+aspect+".csv", index=None)
    scores_history.to_csv("./postprocessing/train_history/scores/scores_history_"+aspect+".csv", index=None)
    
    return model, losses_history, scores_history