import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from numba import njit, prange
from config import CONFIG

def extract_go_terms_and_branches(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        # Match each stanza with [Term] in the OBO file
        stanzas = re.findall(r'\[Term\][\s\S]*?(?=\n\[|$)', content)

    go_terms_dict = {}
    for stanza in stanzas:
        # Extract the GO term ID
        go_id = re.search(r'^id: (GO:\d+)', stanza, re.MULTILINE)
        if go_id:
            go_id = go_id.group(1)

        # Extract the namespace (branch)
        namespace = re.search(r'^namespace: (\w+)', stanza, re.MULTILINE)
        if namespace:
            namespace = namespace.group(1)

        if go_id and namespace:
            # Map the branch abbreviation to the corresponding BPO, CCO, or MFO
            branch_abbr = {'biological_process': 'BPO', 'cellular_component': 'CCO', 'molecular_function': 'MFO'}
            go_terms_dict[go_id] = branch_abbr[namespace]

    return go_terms_dict

def generate_labels_matrix(ids, labels_names, id_labels, go_terms_map):
    labels_matrix = np.zeros((len(ids), len(labels_names)))

    for index, id in tqdm(enumerate(ids)):
        try :
            id_gos_list = id_labels[id]
            temp = [go_terms_map[go] for go in labels_names if go in id_gos_list]
            labels_matrix[index, temp] = 1
        except:
            pass
        
    return labels_matrix

def get_labels_targets(
    ids : np.ndarray,
    labels : pd.DataFrame,
    max_go_terms : int,
    aspect : str,
    ):
    print("GENERATE TARGETS FOR ENTRY IDS ("+str(max_go_terms)+" MAX TOTAL GO TERMS)")

    top_terms = labels.groupby("term")["EntryID"].count().sort_values(ascending=False).to_frame()
    go_terms_aspects=extract_go_terms_and_branches(
        file_path='data/cafa-5-protein-function-prediction/Train/go-basic.obo'
        )
    top_terms["aspect"] = top_terms.index.map(go_terms_aspects)
    print(top_terms[:max_go_terms])
    labels_names = top_terms[:max_go_terms]
    labels_names = labels_names[labels_names.aspect == aspect].index.values
    print("NUMBER OF GO TERMS IN " + aspect + " GROUP :" + str(len(labels_names)))
    train_labels_sub = labels[(labels.term.isin(labels_names)) & (labels.EntryID.isin(ids))]
    id_labels = train_labels_sub.groupby('EntryID')['term'].apply(list).to_dict()
    
    print(len(labels_names))
    go_terms_map = {label: i for i, label in enumerate(labels_names)}
    labels_matrix = generate_labels_matrix(
        ids=ids,
        labels_names=labels_names,
        id_labels=id_labels,
        go_terms_map=go_terms_map
    )
    labels_list = []
    for l in range(labels_matrix.shape[0]):
        labels_list.append(labels_matrix[l, :])

    labels_df = pd.DataFrame(data={"EntryID":ids, "labels_vect":labels_list})
    labels_df.to_pickle("data/train-labels-targets/train_targets_"+aspect+".pkl")
    print("GENERATION FINISHED!")
    return labels_df

def get_aspects(
    aspects_list : list,
    max_go_terms : int
    ): 
    ids = np.load("data/protbert/train_ids.npy")
    labels = pd.read_csv("data/cafa-5-protein-function-prediction/Train/train_terms.tsv", sep = "\t")
    for aspect in aspects_list:
        get_labels_targets(
            ids=ids,
            labels=labels,
            aspect=aspect,
            max_go_terms=max_go_terms
            )