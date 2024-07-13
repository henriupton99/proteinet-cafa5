from config import CONFIG
import gc
import pandas as pd
from inference.predict import make_predictions

sub = pd.DataFrame()
for aspect in CONFIG.ASPECTS:
    temp = make_predictions(
            aspect = aspect,
            prob_threshold = CONFIG.PROB_THRESHOLD,
            embeddings_source = CONFIG.EMBEDDINGS_SOURCE,
            device = CONFIG.DEVICE
            )
    sub = pd.concat([sub, temp])
    del temp
    gc.collect()
    