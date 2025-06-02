# =============================================================================
# Pełny skrypt: temporalny split i wykres „Accuracy vs Step” dla Elliptic
# =============================================================================

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sknetwork.gnn import GNNClassifier
import matplotlib.pyplot as plt

# Jeśli nie masz zainstalowanego sknetwork, odkomentuj poniższą linię i uruchom:
# !pip install scikit-network

# =============================================================================
# 1. Ścieżki do plików (zmodyfikuj, jeśli Twoje pliki leżą gdzie indziej)
# =============================================================================
FEATURES_PATH = 'elliptic_txs_features.csv'
LABELS_PATH   = 'elliptic_txs_classes.csv'
EDGES_PATH    = 'elliptic_txs_edgelist.csv'


# =============================================================================
# 2. Wczytanie cech oraz kroków czasowych
#    ----------------------------------------------------------------------------
#    • Plik elliptic_txs_features.csv nie ma nagłówka.
#      - kolumna 0: tx_id (int)
#      - kolumna 1: step (int od 1 do 49)
#      - kolumny 2..: wektory cech (float)
# =============================================================================
features_df = pd.read_csv(FEATURES_PATH, header=None)
all_tx_ids   = features_df.iloc[:, 0].astype(int).values
all_steps    = features_df.iloc[:, 1].astype(int).values
all_feats    = features_df.iloc[:, 2:].astype(np.float32).values
N_all        = all_tx_ids.shape[0]

# Mapa pomocnicza: tx_id → indeks oryginalny [0..N_all-1]
tx_to_idx_all = {tx: idx for idx, tx in enumerate(all_tx_ids)}

print(f"[INFO] Wczytano {N_all} węzłów (cechy + krok). Przykładowe (tx_id, step, cechy[:3]):")
print(list(zip(all_tx_ids[:5], all_steps[:5], all_feats[:5, :3])), "\n")

# =============================================================================
# 3. Wczytanie etykiet (labels) i usunięcie wierszy „unknown”
#    ----------------------------------------------------------------------------
#    • Plik elliptic_txs_classes.csv ma nagłówek ['txId','class'].
#      W kolumnie 'class' mogą się pojawić wartości „1”, „2” lub „unknown”.
# =============================================================================
labels_df = pd.read_csv(LABELS_PATH, header=0)
print(f"[DEBUG] labels_df.columns przed czyszczeniem: {list(labels_df.columns)}")

# 3.1. Czystka: usuwamy wiersze, w których 'class' jest NaN lub == "unknown"
labels_df = labels_df.dropna(subset=['txId', 'class']).copy()
labels_df = labels_df[labels_df['class'].astype(str).str.lower() != 'unknown'].copy()

# 3.2. Rzutowanie „1”/„2” → int oraz zmiana nazwy kolumny 'txId' → 'tx_id'
labels_df['txId']  = labels_df['txId'].astype(int)
labels_df['class'] = labels_df['class'].astype(int)
labels_df           = labels_df.rename(columns={'txId': 'tx_id'})

# 3.3. Mapowanie etykiet: 1 → 0 (licit), 2 → 1 (illicit)
label_map = {1: 0, 2: 1}
labels_df['label_mapped'] = labels_df['class'].map(label_map)

print(f"[INFO] Po usunięciu „unknown” zostało {len(labels_df)} wierszy etykiet.")
print(f"[DEBUG] labels_df.head():\n{labels_df.head()}\n")

# =============================================================================
# 4. Łączenie etykiet z krokami czasowymi
#    ----------------------------------------------------------------------------
#    • Ponieważ 'step' znajdujemy w features_df (kolumna 1), tworzymy DataFrame:
#        time_df = pd.DataFrame({'tx_id': all_tx_ids, 'step': all_steps})
#    • Następnie robimy merge z labels_df, aby uzyskać labels_time_df z kolumnami:
#        ['tx_id', 'label_mapped', 'step']
# =============================================================================
time_df = pd.DataFrame({'tx_id': all_tx_ids, 'step': all_steps})
labels_time_df = pd.merge(
    labels_df[['tx_id', 'label_mapped']],
    time_df[['tx_id', 'step']],
    on='tx_id',
    how='inner'
)

print(f"[INFO] Po scaleniu etykiet i kroków: {len(labels_time_df)} wierszy.")
print(f"[DEBUG] labels_time_df.head():\n{labels_time_df.head()}\n")

# =============================================================================
# 5. Filtrowanie węzłów do tych, które mają etykietę 0/1
#    ----------------------------------------------------------------------------
#    • Tworzymy zbiór known_tx_ids = {tx_id | ma etykietę 0 lub 1}.
#    • Filtrowanie z features_df według known_tx_ids:
#        filtered_tx_ids, filtered_steps, filtered_feats.
#    • Budujemy:
#        n_nodes = liczba przefiltrowanych węzłów,
#        id_to_idx = {tx_id: lokalny indeks [0..n_nodes-1]}.
#    • Tworzymy wektory:
#        y_all_filtered[i] = etykieta (0/1),
#        step_all_filtered[i] = krok (1..49).
# =============================================================================
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

print(f"[INFO] Po filtrowaniu zostaje {n_nodes} węzłów.")
print(f"[DEBUG] Przykładowe (label, step) pierwszych 10:")
print(list(zip(y_all_filtered[:10], step_all_filtered[:10])), "\n")

# =============================================================================
# 6. Budowa pełnej macierzy adjacency_all
#    ----------------------------------------------------------------------------
#    • Wczytujemy elliptic_txs_edgelist.csv (nagłówek zwykle ['txId1','txId2']).
#    • Zmieniamy nazwy na ['txId_1','txId_2'] w razie potrzeby.
#    • Usuwamy wiersze, gdzie któryś z txId jest NaN.
#    • Filtrujemy krawędzie, bo bierzemy tylko te, gdzie oba txId ∈ id_to_idx.
#    • Mapujemy txId → indeks [0..n_nodes-1], tworzymy macierz COO,
#      dodajemy transpozycję, usuwamy pętle, konwertujemy na CSR.
# =============================================================================
edges_df = pd.read_csv(EDGES_PATH, header=0)
print(f"[DEBUG] edges_df.columns przed rename: {list(edges_df.columns[:5])}")

if 'txId1' in edges_df.columns and 'txId2' in edges_df.columns:
    edges_df = edges_df.rename(columns={'txId1': 'txId_1', 'txId2': 'txId_2'})
elif 'txId_1' in edges_df.columns and 'txId_2' in edges_df.columns:
    pass
else:
    raise ValueError(f"Nieznane nazwy kolumn w pliku edgelist: {list(edges_df.columns)}")

edges_df = edges_df.dropna(subset=['txId_1', 'txId_2']).copy()
mask_edges = edges_df['txId_1'].isin(id_to_idx) & edges_df['txId_2'].isin(id_to_idx)
valid_edges = edges_df[mask_edges].copy()

print(f"[INFO] Wczytano {len(edges_df)} krawędzi, z czego {len(valid_edges)} łączy węzły 0/1.\n")

src = valid_edges['txId_1'].map(id_to_idx).values
dst = valid_edges['txId_2'].map(id_to_idx).values
data    = np.ones(len(src), dtype=np.float32)
adj_coo = coo_matrix((data, (src, dst)), shape=(n_nodes, n_nodes))
adj_sym = adj_coo + adj_coo.T
adj_sym.setdiag(0)
adj_sym.eliminate_zeros()
adjacency_all = adj_sym.tocsr()

print(f"[INFO] Utworzono pełną macierz adjacency_all o kształcie {adjacency_all.shape}.\n")

# =============================================================================
# 7. Pętla po kolejnych wartości T₀ (od 1 do max_step-1) i zbieranie Accuracy
#    ----------------------------------------------------------------------------
max_step = step_all_filtered.max()
steps = []
accuracies = []

for T0 in range(1, max_step):
    # 7.1. Indeksy węzłów treningowych (step ≤ T0) i inferencyjnych (step == T0+1)
    train_time_idx = np.where(step_all_filtered <= T0)[0]
    infer_time_idx = np.where(step_all_filtered == T0 + 1)[0]
    if infer_time_idx.size == 0:
        continue

    # 7.2. Budowa subgrafu trenowanego (T0-only)
    mask_train_nodes = np.zeros(n_nodes, dtype=bool)
    mask_train_nodes[train_time_idx] = True
    adj_train = adjacency_all[mask_train_nodes][:, mask_train_nodes]
    feats_train = filtered_feats[train_time_idx, :]

    # 7.3. Mapowanie „starych” indeksów na nowe lokalne dla subgrafu trenowanego
    new_id_map = {old: new for new, old in enumerate(train_time_idx)}
    labels_train_dict = {new_id_map[idx]: int(y_all_filtered[idx]) for idx in train_time_idx}

    # 7.4. Trening GNNClassifier na subgrafie T0-only
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
        verbose=False
    )
    gnn.fit(adj_train, feats_train, labels_train_dict,
            n_epochs=100, validation=0.1, random_state=42)

    # 7.5. Budowa subgrafu inferencyjnego (T0 + T0+1)
    combined_idx = np.concatenate([train_time_idx, infer_time_idx])
    n_train = train_time_idx.size
    n_infer = infer_time_idx.size
    n_comb = n_train + n_infer
    feats_infer = filtered_feats[combined_idx, :]

    # 7.5.1. Wyciągnięcie pełnej macierzy dla combined_idx i zasłonięcie krawędzi między „nowymi” węzłami
    adj_full = adjacency_all[combined_idx][:, combined_idx].tolil()
    for i_rel in range(n_train, n_comb):
        for j_rel in range(n_train, n_comb):
            adj_full[i_rel, j_rel] = 0.0
    adj_infer = adj_full.tocsr()

    # 7.6. Słownik etykiet dla inferencji: tylko dla węzłów treningowych
    labels_infer_dict = {new_id_map[idx]: int(y_all_filtered[idx]) for idx in train_time_idx}

    # 7.7. Forward pass (n_epochs=0) – model ustawia predykcje w gnn.labels_ dla wszystkich węzłów subgrafu inferencyjnego
    gnn.fit(adj_infer, feats_infer, labels_infer_dict, n_epochs=0, validation=0.0, random_state=42)

    all_pred = gnn.labels_
    pred_new = all_pred[n_train:n_train + n_infer]
    true_new = y_all_filtered[infer_time_idx]
    acc = (pred_new == true_new).sum() / n_infer

    steps.append(T0 + 1)
    accuracies.append(acc)

# =============================================================================
# 8. Rysowanie wykresu „Accuracy vs Step”
#    ----------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(steps, accuracies, marker='o')
plt.xlabel('Step (T0 + 1)')
plt.ylabel('Inference Accuracy')
plt.title('Accuracy vs Step (Temporal Split)')
plt.grid(True)
plt.tight_layout()
plt.show()