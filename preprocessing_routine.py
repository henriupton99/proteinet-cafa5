from preprocessing.targets import generate_targets
from config import CONFIG

generate_targets(
    ids_path = CONFIG.TRAIN_IDS,
    labels_path = CONFIG.TRAIN_LABELS,
    weights_path = CONFIG.IA_WEIGHTS,
    go_obo_path = CONFIG.GO_OBO_FILE,
    evidence_codes_path = CONFIG.EVIDENCE_CODES,
    targets_path = CONFIG.TARGETS_PATH,
    aspects_list = CONFIG.ASPECTS,
    go_terms_per_aspects = CONFIG.ASPECTS_LABELS
)