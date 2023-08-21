from time import time
from config import CONFIG
from training.train import train_model
from time import time

for aspect in CONFIG.ASPECTS:
    print("="*25)
    print("START TRAINING PHASE FOR ASPECT : " , aspect)
    start = time()
    train_model(
        aspect = aspect,
        embeddings_source = CONFIG.EMBEDDINGS_SOURCE,
        num_classes = CONFIG.ASPECTS_LABELS[aspect],
        hidden_dim = CONFIG.ASPECTS_HIDDEN_SIZE[aspect],
        k_folds = CONFIG.K_FOLDS,
        batch_size = CONFIG.BATCHS_SIZE,
        n_epochs = CONFIG.N_EPOCHS,
        learning_rate = CONFIG.LEARNING_RATE,
        device = CONFIG.DEVICE,
        validation_mode = CONFIG.VALIDATION_MODE
    )
    end = time()
    print("TRAINING FINISHED (time elapsed : ", round(end-start,2), " seconds)")