import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sknetwork.gnn import GNNClassifier
import matplotlib.pyplot as plt


FEATURES_PATH = 'elliptic_txs_features.csv'
LABELS_PATH   = 'elliptic_txs_classes.csv'
EDGES_PATH    = 'elliptic_txs_edgelist.csv'

#
features_df = pd.read_csv(FEATURES_PATH, header=None)
all_tx_ids   = features_df.iloc[:, 0].astype(int).values
all_steps    = features_df.iloc[:, 1].astype(int).values
all_feats    = features_df.iloc[:, 2:].astype(np.float32).values  # (N_all, dim_feats)
N_all        = all_tx_ids.shape[0]

# przemapowanie tx_id co by nie przeszukiwać całej listy szukając indeksu
tx_to_idx_all = {tx: idx for idx, tx in enumerate(all_tx_ids)}

labels_df = pd.read_csv(LABELS_PATH, header=0)

# wywaliłem te 'unknown', idk czy coś wnoszą do klasyfikacji no i ten porządek z etykietami co gadaliśmy
labels_df = labels_df.dropna(subset=['txId', 'class']).copy()
labels_df = labels_df[labels_df['class'].astype(str).str.lower() != 'unknown'].copy()

labels_df['txId']  = labels_df['txId'].astype(int)
labels_df['class'] = labels_df['class'].astype(int)
labels_df           = labels_df.rename(columns={'txId': 'tx_id'})

label_map = {1: 0, 2: 1}
labels_df['label_mapped'] = labels_df['class'].map(label_map)

#scalenie etykiez z tymi kolejnymi szeregami 
time_df = pd.DataFrame({'tx_id': all_tx_ids, 'step': all_steps})
labels_time_df = pd.merge(
    labels_df[['tx_id', 'label_mapped']],
    time_df[['tx_id', 'step']],
    on='tx_id',
    how='inner'
)

#nie wiem czy to jest potrzebne jak już wywalone 'unknown', ale zostawiam
known_tx_ids = set(labels_time_df['tx_id'].values)
mask_known   = [tx in known_tx_ids for tx in all_tx_ids]

filtered_tx_ids   = all_tx_ids[mask_known]
filtered_steps    = all_steps[mask_known]
filtered_feats    = all_feats[mask_known, :]
n_nodes           = filtered_tx_ids.shape[0]
id_to_idx = {tx: idx for idx, tx in enumerate(filtered_tx_ids)}

y_all_filtered    = np.zeros(n_nodes, dtype=int)
step_all_filtered = np.zeros(n_nodes, dtype=int)

for _, row in labels_time_df.iterrows():
    tx   = int(row['tx_id'])
    lab  = int(row['label_mapped'])
    stp  = int(row['step'])
    if tx in id_to_idx:
        idx = id_to_idx[tx]
        y_all_filtered[idx]    = lab
        step_all_filtered[idx] = stp


#obrazywanie krawędzi
edges_df = pd.read_csv(EDGES_PATH, header=0)
if 'txId1' in edges_df.columns and 'txId2' in edges_df.columns:
    edges_df = edges_df.rename(columns={'txId1': 'txId_1', 'txId2': 'txId_2'})
elif 'txId_1' in edges_df.columns and 'txId_2' in edges_df.columns:
    pass
else:
    raise ValueError(f"test")

edges_df = edges_df.dropna(subset=['txId_1', 'txId_2']).copy()

mask_edges = edges_df['txId_1'].isin(id_to_idx) & edges_df['txId_2'].isin(id_to_idx)
valid_edges = edges_df[mask_edges].copy()



src = valid_edges['txId_1'].map(id_to_idx).values
dst = valid_edges['txId_2'].map(id_to_idx).values
data    = np.ones(len(src), dtype=np.float32)
adj_coo = coo_matrix((data, (src, dst)), shape=(n_nodes, n_nodes))

adj_sym = adj_coo + adj_coo.T
adj_sym.setdiag(0)
adj_sym.eliminate_zeros()
adjacency_all = adj_sym.tocsr()



T0 = 25    # Maksymalny szereg który trafia do treningu
train_idx = np.where(step_all_filtered <= T0)[0]
n_train = train_idx.size


# tutaj ten 'podgraf robimy'
mask_train_nodes = np.zeros(n_nodes, dtype=bool)
mask_train_nodes[train_idx] = True
adj_train = adjacency_all[mask_train_nodes][:, mask_train_nodes]
feats_train = filtered_feats[train_idx, :]

# Mapowanie starego indeksu na nowy indeks dla treningowych
new_id_map = {old: new for new, old in enumerate(train_idx)}
labels_train_dict = { new_id_map[idx]: int(y_all_filtered[idx]) for idx in train_idx }

gnn = GNNClassifier(
    dims=[64, 32, 2],
    layer_types='Conv',
    activations='ReLu',
    use_bias=True,
    normalizations='both',
    self_embeddings=True,
    loss='CrossEntropy',
    optimizer='Adam',
    learning_rate=0.01,
    early_stopping=True,
    patience=10,
    verbose=True
)
gnn.fit(adj_train, feats_train, labels_train_dict,
        n_epochs=100, validation=0.1, random_state=42)


max_step = step_all_filtered.max()              
steps = list(range(T0 + 1, max_step + 1))
accuracies = []


for k in steps:
    test_idx = np.where(step_all_filtered == k)[0]
    if test_idx.size == 0:
        accuracies.append(np.nan)
        print(f"[INFO] Brak węzłów na kroku {k}, pomijam.")
        continue

    combined_idx = np.concatenate([train_idx, test_idx])
    n_test = test_idx.size
    n_comb = n_train + n_test

    feats_infer = filtered_feats[combined_idx, :]
    adj_full = adjacency_all[combined_idx][:, combined_idx].tolil()

    for i_rel in range(n_train, n_comb):
        for j_rel in range(n_train, n_comb):
            adj_full[i_rel, j_rel] = 0.0

    adj_infer = adj_full.tocsr()

    labels_infer_dict = { new_id_map[idx]: int(y_all_filtered[idx]) for idx in train_idx }
    gnn.fit(adj_infer, feats_infer, labels_infer_dict, n_epochs=0, validation=0.0, random_state=42)

    all_pred = gnn.labels_
    pred_new = all_pred[n_train : n_train + n_test]
    true_new = y_all_filtered[test_idx]
    acc = (pred_new == true_new).sum() / n_test
    accuracies.append(acc)

    print(f"{acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(steps, accuracies, marker='o')
plt.xlabel('Nr. szeregu')
plt.ylabel('acc')
plt.grid(True)
plt.tight_layout()
plt.show()
