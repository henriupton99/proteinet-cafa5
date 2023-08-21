import torch
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from preprocessing.dataset import ProteinSequenceDataset
from training.models import LinearModel

embeds_dim = {
    "T5" : 1024,
    "ProtBERT" : 1024,
    "ESM2" : 2560
}

def make_predictions(
    aspect : str,
    prob_threshold : float,
    embeddings_source : str,
    hidden_size : int,
    device : str
    ):
    """Function for inference from expert model for a given aspect
    NB : This function will be run for each aspect and all predicition datasets will be concatenated fro final submission

    Args:
        aspect (str): aspect name (BPO, CCO, MFO)
        prob_threshold (float): probability threshold to delete too low predictions
        embeddings_source (str): embedding source (ESM2, ProtBERT or T5)
        hidden_size (int): hidden size of layer of expert model for load state dict
        device (str): device for model inference (cuda for GPU, or cpu for CPU)

    Returns:
        submission_df : submission dataset for given aspect
    """
    
    test_dataset = ProteinSequenceDataset(aspect=aspect, datatype="test", embeddings_source = embeddings_source)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    df_weights = pd.read_csv("/kaggle/working/weights_{}.csv".format(aspect))
    labels_names = list(df_weights["term"].values)
    num_labels = len(labels_names)
    
    model = LinearModel(
        input_dim=embeds_dim[embeddings_source],
        hidden_dim=hidden_size,
        num_classes=num_labels).to(device)

    model_path = "/kaggle/input/expert-models-cafa5/{}/expert_model.pt".format(aspect)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("="*25)
    print("GENERATE PREDICTION FOR ASPECT {}".format(aspect))

    ids_ = np.empty(shape=(len(test_dataloader)*num_labels,), dtype=object)
    go_terms_ = np.empty(shape=(len(test_dataloader)*num_labels,), dtype=object)
    confs_ = np.empty(shape=(len(test_dataloader)*num_labels,), dtype=np.float32)

    for i, (embed, id) in tqdm(enumerate(test_dataloader)):
        embed = embed.to(device)
        if device == "cpu":
            confs_[i*num_labels:(i+1)*num_labels] = torch.sigmoid(model(embed)).squeeze().detach().numpy()
        else:
            confs_[i*num_labels:(i+1)*num_labels] = torch.sigmoid(model(embed)).squeeze().detach().cpu().numpy()
        ids_[i*num_labels:(i+1)*num_labels] = id[0]
        go_terms_[i*num_labels:(i+1)*num_labels] = labels_names
    
    len_before_delete = len(ids_)
    rows_to_delete = confs_ < prob_threshold
    confs_ = confs_[~rows_to_delete]
    ids_ = ids_[~rows_to_delete]
    go_terms_ = go_terms_[~rows_to_delete]
    len_after_delete = len(ids_)
    print("NUMBER OF ROWS DELETED (THRESHOLD) : {}".format(len_before_delete - len_after_delete))
    
    del model, rows_to_delete
    gc.collect()
    submission_df = pd.DataFrame(data={"Id" : ids_, "GO term" : go_terms_, "Confidence" : confs_})
    
    del confs_, ids_, go_terms_
    gc.collect()
    submission_df.to_csv("/kaggle/working/predictions_{}.csv".format(aspect))
    print("PREDICTIONS DONE FOR ASPECT {}".format(aspect))
    return submission_df