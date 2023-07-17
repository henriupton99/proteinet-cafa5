from config import CONFIG
from preprocessing import labels_matrix

labels_matrix.get_aspects(
    aspects_list=CONFIG.aspects,
    max_go_terms=CONFIG.max_go_terms
    )