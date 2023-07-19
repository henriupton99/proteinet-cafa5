from config import CONFIG
from inference.predict import makePredictions

makePredictions(
    aspect = "BPO",
    max_go_terms = CONFIG.max_go_terms,
    model_path = "./models/expert_model_BPO.pt" ,
    embeddings_source = "ESM2",
    device = CONFIG.device
    )