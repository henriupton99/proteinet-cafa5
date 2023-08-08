import torch

class CONFIG:
    # CONSTANTS FOR DATA PATHS
    MAIN_DIR = "data/cafa-5-protein-function-prediction"
    GO_OBO_FILE = MAIN_DIR + "/Train/go-basic.obo"
    TRAIN_SEQUENCES_FASTA = MAIN_DIR  + "/Train/train_sequences.fasta"
    TRAIN_LABELS = MAIN_DIR + "/Train/train_terms.tsv"
    TRAIN_IDS = "data/protbert/test_ids.npy"
    TEST_SEQUENCES_FASTA = MAIN_DIR + "/Test (Targets)/testsuperset.fasta"
    TARGETS_PATH = "data/train-labels-targets/"
    

    # CONSTANTS FOR ASPECTS :
    ASPECTS = ["BPO", "CCO", "MFO"]
    ASPECTS_LABELS = {"BPO" : 713, "CCO" : 151, "MFO" : 136}
    MAX_GO_TERMS = 1000

    # CONSTANTS FOR TRAINING : 
    TRAIN_SIZE = 0.9
    N_EPOCHS = 5
    BATCHS_SIZE = 512
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')