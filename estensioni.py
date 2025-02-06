    #------------------------ inizio blocco 0
import torch
import tqdm
import os
import pandas as pd
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch import Tensor, cosine_similarity
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torch_geometric.data import download_url, extract_zip, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero


# Definizione delle classi
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        self.movie_lin = torch.nn.Linear(20, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "user": self.user_emb(data["user"].node_id),
            "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["movie"],
            data["user", "rates", "movie"].edge_label_index,
        )
        return pred

def prepare_data():

    os.environ['TORCH'] = torch.__version__
    
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    extract_zip(download_url(url, '.'), '.')
    
    movies_path = './ml-latest-small/movies.csv'
    ratings_path = './ml-latest-small/ratings.csv'
    
    # Load the entire movie data frame into memory:
    movies_df = pd.read_csv(movies_path, index_col='movieId')

    # Split genres and convert into indicator variables:
    genres = movies_df['genres'].str.get_dummies('|')

    # Use genres as movie input features:
    movie_feat = torch.from_numpy(genres.values).to(torch.float)
    assert movie_feat.size() == (9742, 20)  # 20 genres in total.

    # Load the entire ratings data frame into memory:
    ratings_df = pd.read_csv(ratings_path)

    # Create a mapping from unique user indices to range [0, num_user_nodes):
    unique_user_id = ratings_df['userId'].unique()
    unique_user_id = pd.DataFrame(data={
        'userId': unique_user_id,
        'mappedID': pd.RangeIndex(len(unique_user_id)),
    })
    # Create a mapping from unique movie indices to range [0, num_movie_nodes):
    unique_movie_id = pd.DataFrame(data={
        'movieId': movies_df.index,
        'mappedID': pd.RangeIndex(len(movies_df)),
    })
    # Perform merge to obtain the edges from users and movies:
    ratings_user_id = pd.merge(ratings_df['userId'], unique_user_id,
                                left_on='userId', right_on='userId', how='left')
    ratings_user_id = torch.from_numpy(ratings_user_id['mappedID'].values)
    ratings_movie_id = pd.merge(ratings_df['movieId'], unique_movie_id,
                                left_on='movieId', right_on='movieId', how='left')
    ratings_movie_id = torch.from_numpy(ratings_movie_id['mappedID'].values)

    # With this, we are ready to construct our edge_index in COO format
    # following PyG semantics:
    edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)
    assert edge_index_user_to_movie.size() == (2, 100836)

    data = HeteroData()

    # Save node indices:
    data["user"].node_id = torch.arange(len(unique_user_id))
    data["movie"].node_id = torch.arange(len(movies_df))

    data["movie"].x = movie_feat
    data["user", "rates", "movie"].edge_index = edge_index_user_to_movie

    data = T.ToUndirected()(data)
    assert data.node_types == ["user", "movie"]
    assert data.edge_types == [("user", "rates", "movie"),
                            ("movie", "rev_rates", "user")]
    assert data["user"].num_nodes == 610
    assert data["user"].num_features == 0
    assert data["movie"].num_nodes == 9742
    assert data["movie"].num_features == 20
    assert data["user", "rates", "movie"].num_edges == 100836
    assert data["movie", "rev_rates", "user"].num_edges == 100836

    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        edge_types=("user", "rates", "movie"),
        rev_edge_types=("movie", "rev_rates", "user"),
    )

    train_data, val_data, test_data = transform(data)

    assert train_data["user", "rates", "movie"].num_edges == 56469
    assert train_data["user", "rates", "movie"].edge_label_index.size(1) == 24201
    assert train_data["movie", "rev_rates", "user"].num_edges == 56469
    # No negative edges added:
    assert train_data["user", "rates", "movie"].edge_label.min() == 1
    assert train_data["user", "rates", "movie"].edge_label.max() == 1

    assert val_data["user", "rates", "movie"].num_edges == 80670
    assert val_data["user", "rates", "movie"].edge_label_index.size(1) == 30249
    assert val_data["movie", "rev_rates", "user"].num_edges == 80670
    # Negative edges with ratio 2:1:
    assert val_data["user", "rates", "movie"].edge_label.long().bincount().tolist() == [20166, 10083]

    return train_data, val_data, test_data

# Funzione per estrarre embedding e fare Collaborative Filtering
def recommend_movies(user_embeddings, movie_embeddings, top_k=5, batch_size=100):
    user_embeddings = user_embeddings.cpu()
    movie_embeddings = movie_embeddings.cpu()
    recommendations = {}

    for start in range(0, len(user_embeddings), batch_size):
        end = min(start + batch_size, len(user_embeddings))
        similarity_matrix = F.cosine_similarity(user_embeddings[start:end].unsqueeze(1),
                                                movie_embeddings.unsqueeze(0), dim=-1)
        for i, user_idx in enumerate(range(start, end)):
            recommended_movie_indices = similarity_matrix[i].argsort(descending=True)[:top_k]
            recommendations[user_idx] = recommended_movie_indices.tolist()
    
    return recommendations

#------------------------ fine blocco 0 e inzio blocco 1
def train_and_evaluate(hidden_channels, learning_rate, batch_size, num_neighbors, neg_sampling_ratio, excel_filename):
    train_data, val_data, test_data = prepare_data()

    # In the first hop, we sample at most 20 neighbors.
    # In the second hop, we sample at most 10 neighbors.
    # In addition, during training, we want to sample negative edges on-the-fly with
    # a ratio of 2:1.
    # We can make use of the loader.LinkNeighborLoader from PyG:
    

    # Define seed edges:
    edge_label_index = train_data["user", "rates", "movie"].edge_label_index
    edge_label = train_data["user", "rates", "movie"].edge_label

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=num_neighbors,
        neg_sampling_ratio=neg_sampling_ratio,
        edge_label_index=(("user", "rates", "movie"), edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=True,
    )

    # Inspect a sample:
    sampled_data = next(iter(train_loader))

    model = Model(hidden_channels=hidden_channels , data=train_data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    for epoch in range(1, 6):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data["user", "rates", "movie"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            loss = total_loss / total_examples
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

    # Define the validation seed edges:
    edge_label_index = val_data["user", "rates", "movie"].edge_label_index
    edge_label = val_data["user", "rates", "movie"].edge_label

    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[20, 10],
        edge_label_index=(("user", "rates", "movie"), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * 128,
        shuffle=False,
    )

    sampled_data = next(iter(val_loader))


    assert sampled_data["user", "rates", "movie"].edge_label_index.size(1) == 3 * 128
    assert sampled_data["user", "rates", "movie"].edge_label.min() >= 0
    assert sampled_data["user", "rates", "movie"].edge_label.max() <= 1

    preds = []
    ground_truths = []
    total_val_loss = total_val_examples = 0 
    for sampled_data in tqdm.tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            val_loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            total_val_loss += float(val_loss) * pred.numel()
            total_val_examples += pred.numel()
            preds.append(model(sampled_data))
            ground_truths.append(sampled_data["user", "rates", "movie"].edge_label)

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    print()
    print(f"Validation AUC: {auc:.4f}")


    # inserite da noi

    f1 = f1_score(ground_truth, (pred > 0.5).astype(int))  # soglia a 0.5 per determinare la classificazione
    print(f"Validation F1 Score: {f1:.4f}")

    precision = precision_score(ground_truth, (pred > 0.5).astype(int))
    print(f"Validation Precision: {precision:.4f}")

    recall = recall_score(ground_truth, (pred > 0.5).astype(int))
    print(f"Validation Recall: {recall:.4f}")
    
    avg_loss = total_val_loss / total_val_examples
    print(f"Validation Loss: {avg_loss:.4f}")
    # inserisci nel file i valori
    
    experiment_data = {
        "Hidden Channels": [hidden_channels],
        "Learning Rate": [learning_rate],
        "Batch Size": [batch_size],
        "Num Neighbors": [str(num_neighbors)],
        "Neg Sampling Ratio": [neg_sampling_ratio],
        "AUC": [auc],
        "F1-score": [f1],
        "Precision": [precision],
        "Recall": [recall],
        "Loss": [avg_loss],
    }
        
    try:
        df_existing = pd.read_excel(excel_filename)
        df = pd.concat([df_existing, pd.DataFrame(experiment_data)], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame(experiment_data)

    try:
        df.to_excel(excel_filename, index=False)
    except PermissionError:
        print(f"Permission denied: '{excel_filename}'. Please close the file if it is open in another program.")
        
    df.to_excel(excel_filename, index=False)
    print(f"Risultati salvati in {excel_filename}")
    
    # Estrai gli embedding degli utenti e dei film
    user_embeddings = model.user_emb.weight.data
    movie_embeddings = model.movie_emb.weight.data

    # Genera raccomandazioni per ogni utente
    recommendations = recommend_movies(user_embeddings, movie_embeddings, top_k=5)

    # Simuliamo che l'utente accetti alcuni consigli (prendiamo i primo film suggeriti per ogni utente)
    simulated_feedback = {user: recs[:1] for user, recs in recommendations.items()}

    MAX_HISTORY = 20  # Numero massimo di feedback per utente

    # Dizionario per tenere traccia delle interazioni utente-film già presenti
    user_feedback = {}

    # Recupera le interazioni esistenti nel dataset
    existing_edges = train_data["user", "rates", "movie"].edge_index.cpu().T.tolist()

    # Popola il dizionario con le interazioni esistenti
    for user, movie in existing_edges:
        if user not in user_feedback:
            user_feedback[user] = []
        user_feedback[user].append(movie)

    # Aggiunge solo le nuove interazioni, mantenendo al massimo MAX_HISTORY per utente
    for user, movies in simulated_feedback.items():
        if user not in user_feedback:
            user_feedback[user] = []
        
        user_feedback[user].extend(movies)  # Aggiunge i nuovi film consigliati
        user_feedback[user] = user_feedback[user][-MAX_HISTORY:]  # Mantiene solo gli ultimi 5 film

    # Convertiamo il nuovo set di interazioni in tensore
    filtered_edges = [[user, movie] for user, movies in user_feedback.items() for movie in movies]
    filtered_edges = torch.tensor(filtered_edges).T  # Shape (2, num_interazioni_filtrate)

    # Aggiorna il dataset con le nuove interazioni "filtrate"
    train_data["user", "rates", "movie"].edge_index = filtered_edges



    
    #---------------- fine blocco 12 e inzio blocco 13
    

# codice che esegue la funzione train_and_evaluate per ogni valore di iperpametri con cicli for
# e salva i risultati su un file excel

# Creazione del file Excel
excel_filename = "resultsAdamWExtension25.xlsx"
df = pd.DataFrame(columns=["Hidden Channels", "Learning Rate", "Batch Size", "Num Neighbors", "Neg Sampling Ratio", "AUC", "F1-score", "Precision", "Recall", "Loss"])
df.to_excel(excel_filename, index=False)

# Richiamo della funzione con vari iperparametri
hidden_channels_list = [32, 64, 128, 256]
learning_rates = [0.01, 0.005, 0.001, 0.0005]
batch_sizes = [64, 128, 256]
num_neighbors_list = [[10, 5], [20, 10], [30, 15]]
neg_sampling_ratios = [1.0, 2.0, 3.0]

for hidden_channels in hidden_channels_list:
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for num_neighbors in num_neighbors_list:
                for neg_sampling_ratio in neg_sampling_ratios:
                    train_and_evaluate(hidden_channels, learning_rate, batch_size, num_neighbors, neg_sampling_ratio, excel_filename)
                    print(f"hidden_channels: {hidden_channels}, learning_rate: {learning_rate}, batch_size: {batch_size}, num_neighbors: {num_neighbors}, neg_sampling_ratio: {neg_sampling_ratio}")
                    print("-----------------------------------------------------------------------------------------------------------------")           

#---------------- fine blocco 13 e fine file