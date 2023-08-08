from config import CONFIG
import gc
import pandas as pd
from inference.predict import make_predictions

sub = pd.DataFrame()
for aspect in ["BPO", "CCO", "MFO"]:
    temp =make_predictions(
            aspect = "BPO",
            max_go_terms = CONFIG.max_go_terms,
            model_path = "./models/expert_model_BPO.pt" ,
            embeddings_source = "ESM2",
            device = CONFIG.device
            )
    sub = pd.concat([sub, temp])
    del temp
    gc.collect()
    