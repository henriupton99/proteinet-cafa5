from preprocessing.targets import generate_targets
from config import CONFIG

generate_targets(
    ids_path = CONFIG.TRAIN_IDS,
    labels_path = CONFIG.TRAIN_LABELS,
    go_obo_path = CONFIG.GO_OBO_FILE,
    save_path = CONFIG.TARGETS_PATH,
    max_go_terms = CONFIG.MAX_GO_TERMS,
    aspects_list = CONFIG.ASPECTS
    )