from time import time
from config import CONFIG
from preprocessing import labels_matrix
from training.train import train_model

for aspect in CONFIG.aspects:
    print("************************")
    print("START TRAINING PHASE FOR ASPECT : " , aspect)
    start = time()
    train_model(
        aspect=aspect,
        embeddings_source="ESM2",
        model_type="linear",
        num_classes=CONFIG.aspects_num_labels[aspect],
        train_size=CONFIG.train_size,
        batch_size=CONFIG.batch_size,
        n_epochs=CONFIG.n_epochs,
        lr=CONFIG.lr,
        device=CONFIG.device
    )
    end = time()
    print("TRAINING FINISHED (time elapsed : ", round(end-start,2), " seconds)")
    print("¬¬¬¬ MODEL SAVED IN THE /models FOLDER ¬¬¬¬")
    print("************************")