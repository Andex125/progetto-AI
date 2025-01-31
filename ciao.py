    #------------------------ inizio blocco 1
import torch
from torch import Tensor
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import tqdm
import os
from torch_geometric.data import download_url, extract_zip, HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F

def train_and_evaluate(hidden_channels, learning_rate, batch_size, num_neighbors, neg_sampling_ratio, excel_filename):

    print(torch.__version__)

    #------------------------ fine blocco 1 e inzio blocco 2

    os.environ['TORCH'] = torch.__version__

    #!pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
    #!pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
    #!pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}.html
    #!pip install git+https://github.com/pyg-team/pytorch_geometric.git

    #--------------------- fine blocco 2 e inzio blocco 3

    from torch_geometric.data import download_url, extract_zip

    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    extract_zip(download_url(url, '.'), '.')

    movies_path = './ml-latest-small/movies.csv'
    ratings_path = './ml-latest-small/ratings.csv'

    #--------------- fine blocco 3 e e inzio blocco 4

    import pandas as pd

    print('movies.csv:')
    print('===========')
    print(pd.read_csv(movies_path)[["movieId", "genres"]].head())
    print()
    print('ratings.csv:')
    print('============')
    print(pd.read_csv(ratings_path)[["userId", "movieId"]].head())

    #------------------- fine blocco 4 e e inzio blocco 5

    # Load the entire movie data frame into memory:
    movies_df = pd.read_csv(movies_path, index_col='movieId')

    # Split genres and convert into indicator variables:
    genres = movies_df['genres'].str.get_dummies('|')
    print(genres[["Action", "Adventure", "Drama", "Horror"]].head())

    # Use genres as movie input features:
    movie_feat = torch.from_numpy(genres.values).to(torch.float)
    assert movie_feat.size() == (9742, 20)  # 20 genres in total.

    #------------------------ fine blocco 5 e inzio blocco 6

    # Load the entire ratings data frame into memory:
    ratings_df = pd.read_csv(ratings_path)

    # Create a mapping from unique user indices to range [0, num_user_nodes):
    unique_user_id = ratings_df['userId'].unique()
    unique_user_id = pd.DataFrame(data={
        'userId': unique_user_id,
        'mappedID': pd.RangeIndex(len(unique_user_id)),
    })
    print("Mapping of user IDs to consecutive values:")
    print("==========================================")
    print(unique_user_id.head())
    print()
    # Create a mapping from unique movie indices to range [0, num_movie_nodes):
    unique_movie_id = pd.DataFrame(data={
        'movieId': movies_df.index,
        'mappedID': pd.RangeIndex(len(movies_df)),
    })
    print("Mapping of movie IDs to consecutive values:")
    print("===========================================")
    print(unique_movie_id.head())

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

    print()
    print("Final edge indices pointing from users to movies:")
    print("=================================================")
    print(edge_index_user_to_movie)

    #----------------- fine blocco 6 e inzio blocco 7


    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T

    data = HeteroData()

    # Save node indices:
    data["user"].node_id = torch.arange(len(unique_user_id))
    data["movie"].node_id = torch.arange(len(movies_df))

    # Add the node features and edge indices:
    #data["movie"].x = ...  # TODO
    #data["user", "rates", "movie"].edge_index = ...  # TODO
    data["movie"].x = movie_feat
    data["user", "rates", "movie"].edge_index = edge_index_user_to_movie

    # We also need to make sure to add the reverse edges from movies to users
    # in order to let a GNN be able to pass messages in both directions.
    # We can leverage the T.ToUndirected() transform for this from PyG:

    # TODO:
    #raise NotImplementedError
    data = T.ToUndirected()(data)

    print(data)

    assert data.node_types == ["user", "movie"]
    assert data.edge_types == [("user", "rates", "movie"),
                            ("movie", "rev_rates", "user")]
    assert data["user"].num_nodes == 610
    assert data["user"].num_features == 0
    assert data["movie"].num_nodes == 9742
    assert data["movie"].num_features == 20
    assert data["user", "rates", "movie"].num_edges == 100836
    assert data["movie", "rev_rates", "user"].num_edges == 100836

    #------------------ fine  blocco 7 e inzio blocco 8

    # For this, we first split the set of edges into
    # training (80%), validation (10%), and testing edges (10%).
    # Across the training edges, we use 70% of edges for message passing,
    # and 30% of edges for supervision.
    # We further want to generate fixed negative edges for evaluation with a ratio of 2:1.
    # Negative edges during training will be generated on-the-fly, so we don't want to
    # add them to the graph right away.
    # Overall, we can leverage the RandomLinkSplit() transform for this from PyG:
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        edge_types=("user", "rates", "movie"),
        rev_edge_types=("movie", "rev_rates", "user"),
        #num_val=...,  # TODO
        #num_test=...,  # TODO
        #disjoint_train_ratio=...,  # TODO
        #neg_sampling_ratio=...,  # TODO
        #add_negative_train_samples=...,  # TODO
        #edge_types=("user", "rates", "movie"),
        #rev_edge_types=("movie", "rev_rates", "user"),
    )

    train_data, val_data, test_data = transform(data)
    print("Training data:")
    print("==============")
    print(train_data)
    print()
    print("Validation data:")
    print("================")
    print(val_data)

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

    #---------------- fine blocco 8 e e inzio blocco 9

    # In the first hop, we sample at most 20 neighbors.
    # In the second hop, we sample at most 10 neighbors.
    # In addition, during training, we want to sample negative edges on-the-fly with
    # a ratio of 2:1.
    # We can make use of the loader.LinkNeighborLoader from PyG:
    from torch_geometric.loader import LinkNeighborLoader

    # Define seed edges:
    edge_label_index = train_data["user", "rates", "movie"].edge_label_index
    edge_label = train_data["user", "rates", "movie"].edge_label

    train_loader = LinkNeighborLoader(
        #data=...,  # TODO
        #num_neighbors=...,  # TODO
        #neg_sampling_ratio=...,  # TODO
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

    print("Sampled mini-batch:")
    print("===================")
    print(sampled_data)

    #assert sampled_data["user", "rates", "movie"].edge_label_index.size(1) == 3 * 128
    #assert sampled_data["user", "rates", "movie"].edge_label.min() == 0
    #assert sampled_data["user", "rates", "movie"].edge_label.max() == 1

    #--------------- fine blocco 9 e inzio blocco 10

    from torch_geometric.nn import SAGEConv, to_hetero
    #TODO
    import torch.nn.functional as F

    class GNN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()

            self.conv1 = SAGEConv(hidden_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
            # Define a 2-layer GNN computation graph.
            # Use a *single* ReLU non-linearity in-between.
            # TODO:
            #raise NotImplementedError
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return x

    # Our final classifier applies the dot-product between source and destination
    # node embeddings to derive edge-level predictions:
    class Classifier(torch.nn.Module):
        def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
            # Convert node embeddings to edge-level representations:
            edge_feat_user = x_user[edge_label_index[0]]
            edge_feat_movie = x_movie[edge_label_index[1]]

            # Apply dot-product to get a prediction per supervision edge:
            return (edge_feat_user * edge_feat_movie).sum(dim=-1)


    class Model(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            # Since the dataset does not come with rich features, we also learn two
            # embedding matrices for users and movies:
            self.movie_lin = torch.nn.Linear(20, hidden_channels)
            self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
            self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)

            # Instantiate homogeneous GNN:
            self.gnn = GNN(hidden_channels)

            # Convert GNN model into a heterogeneous variant:
            self.gnn = to_hetero(self.gnn, metadata=data.metadata())

            self.classifier = Classifier()

        def forward(self, data: HeteroData) -> Tensor:
            x_dict = {
            "user": self.user_emb(data["user"].node_id),
            "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
            }

            # x_dict holds feature matrices of all node types
            # edge_index_dict holds all edge indices of all edge types
            x_dict = self.gnn(x_dict, data.edge_index_dict)

            pred = self.classifier(
                x_dict["user"],
                x_dict["movie"],
                data["user", "rates", "movie"].edge_label_index,
            )

            return pred


    model = Model(hidden_channels=hidden_channels)

    print(model)

    #----------- fine blocco 10 e inzio blocco 11

    import tqdm
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(1, 6):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()

            # TODO: Move sampled_data to the respective device
            # TODO: Run forward pass of the model
            # TODO: Apply binary cross entropy via
            # F.binary_cross_entropy_with_logits(pred, ground_truth)
            # raise NotImplementedError

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

    # ----------- fine blocco 11 e inizio 12
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

    print("Sampled mini-batch:")
    print("===================")
    print(sampled_data)

    assert sampled_data["user", "rates", "movie"].edge_label_index.size(1) == 3 * 128
    assert sampled_data["user", "rates", "movie"].edge_label.min() >= 0
    assert sampled_data["user", "rates", "movie"].edge_label.max() <= 1

    # ----------- fine blocco 12 e inzio blocco 13

    from sklearn.metrics import roc_auc_score

    preds = []
    ground_truths = []
    total_val_loss = total_val_examples = 0 
    for sampled_data in tqdm.tqdm(val_loader):
        with torch.no_grad():
            # TODO: Collect predictions and ground-truths and write them into
            # preds and ground_truths.
            # raise NotImplementedError
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
    

    
    # ---------- fine blocco 12
    

# codice che esegue la funzione train_and_evaluate per ogni valore di iperpametri con cicli for
# e salva i risultati su un file excel
# Creazione del file Excel
excel_filename = "results.xlsx"
df = pd.DataFrame(columns=["Hidden Channels", "Learning Rate", "Batch Size", "Num Neighbors", "Neg Sampling Ratio", "AUC", "F1-score", "Precision", "Recall", "Loss"])
df.to_excel(excel_filename, index=False)

# Richiamo della funzione con vari iperparametri
hidden_channels_list = [32, 64, 128, 256]
learning_rates = [0.01, 0.005, 0.001, 0.0005]
batch_sizes = [64, 128, 256]
num_neighbors_list = [[10, 5], [20, 10], [30, 15]]
neg_sampling_ratios = [1.0, 2.0, 3.0]

#testing se funzionava
#hidden_channels_list = [32]
#learning_rates = [0.01]
#batch_sizes = [64]
#num_neighbors_list = [[10, 5]]
#neg_sampling_ratios = [1.0, 2.0, 3.0]

for hidden_channels in hidden_channels_list:
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for num_neighbors in num_neighbors_list:
                for neg_sampling_ratio in neg_sampling_ratios:
                    train_and_evaluate(hidden_channels, learning_rate, batch_size, num_neighbors, neg_sampling_ratio, excel_filename)
                    print(f"hidden_channels: {hidden_channels}, learning_rate: {learning_rate}, batch_size: {batch_size}, num_neighbors: {num_neighbors}, neg_sampling_ratio: {neg_sampling_ratio}")
                    print("-----------------------------------------------------------------------------------------------------------------")           

