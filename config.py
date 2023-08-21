import torch

# CONSTANTS FOR DATA PATHS
MAIN_DIR = "/kaggle/input/cafa-5-protein-function-prediction/"
GO_OBO_FILE = MAIN_DIR + "Train/go-basic.obo"
TRAIN_SEQUENCES_FASTA = MAIN_DIR  + "Train/train_sequences.fasta"
TRAIN_LABELS = MAIN_DIR + "Train/train_terms.tsv"
TRAIN_IDS = "/kaggle/input/protbert-embeddings-for-cafa5/train_ids.npy"
IA_WEIGHTS = MAIN_DIR + "IA.txt"
TEST_SEQUENCES_FASTA = MAIN_DIR + "/Test (Targets)/testsuperset.fasta"
TARGETS_PATH = "/kaggle/working/train-labels-targets/"
EVIDENCE_CODES = "/kaggle/input/enhanced-train-terms/propagated_evidenceCode.parquet"

# CONSTANTS FOR ASPECTS :
ASPECTS = ["BPO", "CCO", "MFO"]
ASPECTS_LABELS = {"BPO" : 1100, "CCO" : 300, "MFO" : 450}

# CONSTANTS FOR TRAINING : 
EMBEDDINGS_SOURCE = "ESM2"
K_FOLDS = 5
N_EPOCHS = 15
BATCHS_SIZE = 256
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ASPECTS_HIDDEN_SIZE = {"BPO" : 1256, "CCO" : 512, "MFO" : 850}
VALIDATION_MODE = False

# CONSTANTS FOR POSTPROCESSING :
PROB_THRESHOLD = 0.10