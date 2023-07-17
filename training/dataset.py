import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ProteinSequenceDataset(Dataset):
    
    embeds_map = {
    "T5" : "t5",
    "ProtBERT" : "protbert",
    "ESM2" : "esm2"
    }
    
    def __init__(self, aspect, datatype, embeddings_source):
        super(ProteinSequenceDataset).__init__()
        self.datatype = datatype
        
        embeds = np.load("../data/"+ProteinSequenceDataset.embeds_map[embeddings_source]+"/"+datatype+"_embeddings.npy")
        ids = np.load("../data/"+ProteinSequenceDataset.embeds_map[embeddings_source]+"/"+datatype+"_ids.npy")
            
        embeds_list = []
        for l in range(embeds.shape[0]):
            embeds_list.append(embeds[l,:])
        self.df = pd.DataFrame(data={"EntryID": ids, "embed" : embeds_list})
        
        if datatype=="train":
            df_labels = pd.read_pickle(
                "../data/train-labels-targets/train_targets_"+aspect+".pkl")
            self.df = self.df.merge(df_labels, how="right", on="EntryID")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        embed = torch.tensor(self.df.iloc[index]["embed"] , dtype = torch.float32)
        if self.datatype=="train":
            targets = torch.tensor(self.df.iloc[index]["labels_vect"], dtype = torch.float32)
            return embed, targets
        if self.datatype=="test":
            id = self.df.iloc[index]["EntryID"]
            return embed, id
        
dataset = ProteinSequenceDataset(aspect = "BPO", datatype = "train", embeddings_source = "ESM2")

print(dataset.__getitem__(0))