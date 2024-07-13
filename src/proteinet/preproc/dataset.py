import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ProteinSequenceDataset(Dataset):
    
    def __init__(
        self, 
        aspect: str, 
        datatype: str, 
        embeddings_path: str, 
        embeddings_source: str,
        targets_path: str
        ):
        super(ProteinSequenceDataset).__init__()
        self.datatype = datatype
        
        embeds = np.load(os.path.join(embeddings_path, f'{datatype}_embeddings.npy'))
        embeds = [embeds[l,:] for l in range(embeds.shape[0])]
        ids = np.load(os.path.join(embeddings_path, f'{datatype}_ids.npy'))
            
        self.df = pd.DataFrame(data={"EntryID": ids, "embed" : embeds})
        
        if datatype=='train':
            df_labels = pd.read_pickle(os.path.join(targets_path, f'train_targets_{aspect}.pkl'))
            self.df = self.df.merge(df_labels, how="right", on="EntryID")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(
        self, 
        index: int
        ):
        embed = torch.tensor(self.df.iloc[index]["embed"] , dtype = torch.float32)
        if self.datatype=='train':
            targets = torch.tensor(self.df.iloc[index]["labels_vect"], dtype = torch.float32)
            return embed, targets
        if self.datatype=='test':
            id = self.df.iloc[index]["EntryID"]
            return embed, id