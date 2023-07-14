def get_labels_targets(aspect : str):
    print("GENERATE TARGETS FOR ENTRY IDS ("+str(config.max_go_terms)+" MAX TOTAL GO TERMS)")
    ids = np.load("/kaggle/input/protbert-embeddings-for-cafa5/train_ids.npy")
    labels = pd.read_csv(config.train_labels_path, sep = "\t")

    top_terms = labels.groupby("term")["EntryID"].count().sort_values(ascending=False).to_frame()
    top_terms["aspect"] = top_terms.index.map(go_terms_aspects)
    print(top_terms[:config.max_go_terms])
    labels_names = top_terms[:config.max_go_terms]
    labels_names = labels_names[labels_names.aspect == aspect].index.values
    print("NUMBER OF GO TERMS IN " + aspect + " GROUP :" + str(len(labels_names)))
    train_labels_sub = labels[(labels.term.isin(labels_names)) & (labels.EntryID.isin(ids))]
    id_labels = train_labels_sub.groupby('EntryID')['term'].apply(list).to_dict()
    
    print(len(labels_names))
    go_terms_map = {label: i for i, label in enumerate(labels_names)}
    labels_matrix = np.zeros((len(ids), len(labels_names)))

    for index, id in tqdm(enumerate(ids)):
        try :
            id_gos_list = id_labels[id]
            temp = [go_terms_map[go] for go in labels_names if go in id_gos_list]
            labels_matrix[index, temp] = 1
        except:
            pass

    labels_list = []
    for l in range(labels_matrix.shape[0]):
        labels_list.append(labels_matrix[l, :])

    labels_df = pd.DataFrame(data={"EntryID":ids, "labels_vect":labels_list})
    labels_df.to_pickle("/kaggle/working/train_targets_"+aspect+".pkl")
    print("GENERATION FINISHED!")
    return labels_df

for aspect in config.aspects:
    get_labels_targets(aspect)