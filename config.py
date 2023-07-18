import torch

class CONFIG:
    
    MAIN_DIR = "data/cafa-5-protein-function-prediction"
    train_sequences_path = MAIN_DIR  + "/Train/train_sequences.fasta"
    train_labels_path = MAIN_DIR + "/Train/train_terms.tsv"
    test_sequences_path = MAIN_DIR + "/Test (Targets)/testsuperset.fasta"
    ia_path = MAIN_DIR + "/IA.txt"
    
    aspects = ["BPO", "CCO", "MFO"]
    aspects_num_labels = {"BPO" : 713, "CCO" : 151, "MFO" : 136}
    max_go_terms = 1000
    
    train_size = 0.9
    n_epochs = 5
    batch_size = 512
    lr = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')