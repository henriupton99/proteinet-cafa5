import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from preprocessing.dataset import ProteinSequenceDataset
from preprocessing.targets import extract_go_terms_and_branches
from training.models import LinearModel

embeds_dim = {
    "T5" : 1024,
    "ProtBERT" : 1024,
    "ESM2" : 2560
}


def make_predictions(
    aspect,
    max_go_terms,
    model_path,
    embeddings_source,
    device
    ):
    
    test_dataset = ProteinSequenceDataset(aspect=aspect, datatype="test", embeddings_source = embeddings_source)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    labels = pd.read_csv("data/cafa-5-protein-function-prediction/Train/train_terms.tsv", sep = "\t")
    top_terms = labels.groupby("term")["EntryID"].count().sort_values(ascending=False).to_frame()
    go_terms_aspects=extract_go_terms_and_branches(
        file_path='data/cafa-5-protein-function-prediction/Train/go-basic.obo'
        )
    top_terms["aspect"] = top_terms.index.map(go_terms_aspects)
    print(top_terms[:max_go_terms])
    labels_names = top_terms[:max_go_terms]
    labels_names = labels_names[labels_names.aspect == aspect].index.values
    num_labels = len(labels_names)
    print(num_labels)
    print(labels_names)
    
    model = LinearModel(input_dim=embeds_dim[embeddings_source], num_classes=num_labels).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("GENERATE PREDICTION FOR TEST SET...")

    ids_ = np.empty(shape=(len(test_dataloader)*num_labels,), dtype=object)
    go_terms_ = np.empty(shape=(len(test_dataloader)*num_labels,), dtype=object)
    confs_ = np.empty(shape=(len(test_dataloader)*num_labels,), dtype=np.float32)

    for i, (embed, id) in tqdm(enumerate(test_dataloader)):
        embed = embed.to(device)
        confs_[i*num_labels:(i+1)*num_labels] = torch.sigmoid(model(embed)).squeeze().detach().cpu().numpy()
        ids_[i*num_labels:(i+1)*num_labels] = id[0]
        go_terms_[i*num_labels:(i+1)*num_labels] = labels_names

    submission_df = pd.DataFrame(data={"Id" : ids_, "GO term" : go_terms_, "Confidence" : confs_})
    submission_df.to_csv("./data/predictions/predictions_"+aspect+".csv")
    print("PREDICTIONS DONE")
    return submission_df