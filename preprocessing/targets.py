"""FUNCTIONS FOR GENERATING THE LABELS TAGRETS FOR ALL ASPECTS
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

def extract_go_terms_and_branches(
    file_path : str
    ) -> dict:
    """Utilitary function to construct a mapping {GO TERM : ASPECT} for each GO TERM in input OBO file

    Args:
        file_path (str): file path for input GO terms

    Returns:
        go_terms_dict (dict): mapping dictionnary
    """
    with open(file_path, 'r') as file:
        content = file.read()
        stanzas = re.findall(r'\[Term\][\s\S]*?(?=\n\[|$)', content)

    go_terms_dict = {}
    for stanza in stanzas:
        go_id = re.search(r'^id: (GO:\d+)', stanza, re.MULTILINE)
        if go_id:
            go_id = go_id.group(1)

        namespace = re.search(r'^namespace: (\w+)', stanza, re.MULTILINE)
        if namespace:
            namespace = namespace.group(1)

        if go_id and namespace:
            branch_abbr = {'biological_process': 'BPO', 'cellular_component': 'CCO', 'molecular_function': 'MFO'}
            go_terms_dict[go_id] = branch_abbr[namespace]

    return go_terms_dict

def generate_labels_matrix(
    ids : np.ndarray,
    labels_names : list[str],
    id_labels : dict,
    go_terms_map : dict
    ):
    """Utilitary function to generate labels target matrix given :
    - protein ids, labels_names (GO terms names)
    - id labels : id of labels
    - go terms map generated by function extract_go_terms_and_branches

    """
    labels_matrix = np.zeros((len(ids), len(labels_names)))

    for index, id in tqdm(enumerate(ids)):
        try :
            id_gos_list = id_labels[id]
            temp = [go_terms_map[go] for go in labels_names if go in id_gos_list]
            labels_matrix[index, temp] = 1
        except:
            pass
        
    return labels_matrix

def generate_targets(
    ids_path : str,
    labels_path : str,
    go_obo_path : str,
    targets_path : str,
    max_go_terms : int,
    aspects_list : str,
    ):
    """Function to generate labels (targets) for a given aspect (BPO, CCO, or MFO) for each protein id in ids, based on labels dataframe.
    NB : For memory usage and models precision reasons, we only consider a subset of all GO terms labels
    We consider to top K most present (based on max_go_term input number)

    Args:
        ids_path (str): path to protein ids 
        labels_paths (str): path to labels annotations dataframe for each protein in ids
        go_obo_path (str) : path to obo graph file
        targets_path (str) : path where to save the target labels
        max_go_terms (int): number of GO terms to consider (top K most frequent)
        aspects_list (str): list of aspects to consider

    Returns:
        labels_df : dataframe that gather proteins ids and their respective labels vector
    """
    ids = np.load(ids_path)
    labels = pd.read_csv(labels_path, sep = "\t")
    print(labels)
    print("GENERATE TARGETS FOR ENTRY IDS (" + str(max_go_terms) + " MAX TOTAL GO TERMS)")
    for aspect in aspects_list:
        top_terms = labels.groupby("term")["EntryID"].count().sort_values(ascending=False).to_frame()
        map_go_terms_aspects=extract_go_terms_and_branches(
            file_path=go_obo_path
            )
        top_terms["aspect"] = top_terms.index.map(map_go_terms_aspects)
        print(top_terms[:max_go_terms])
        labels_names = top_terms[:max_go_terms]
        labels_names = labels_names[labels_names.aspect == aspect].index.values
        print("NUMBER OF GO TERMS IN " + aspect + " GROUP :" + str(len(labels_names)))
        train_labels_sub = labels[(labels.term.isin(labels_names)) & (labels.EntryID.isin(ids))]
        id_labels = train_labels_sub.groupby('EntryID')['term'].apply(list).to_dict()
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
        labels_df.to_pickle(targets_path + "train_targets_"+ aspect +".pkl")
        print("GENERATION FINISHED!")