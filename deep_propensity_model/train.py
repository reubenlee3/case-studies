import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


parser = argparse.ArgumentParser(description="Test run")

# Add arguments
parser.add_argument(
    "--data_dir",
    required=True,
    help="path to directory"
)

parser.add_argument(
    "--user_history_length",
    required=True,
    help="eg 5 ratings to look back on"
)

parser.add_argument(
    "--batch_size",
    required=True,
    help="batch size 16, 32, 64, 256 etc"
)

parser.add_argument(
    "--user_embedding_size",
    required=True,
    help="64, 128, etc"
)

parser.add_argument(
    "--product_embedding_size",
    required=True,
    help="64, 128, etc"
)
parser.add_argument(
    "--hidden_size",
    required=True,
    help="64, 128, etc"
)

parser.add_argument(
    "--num_epochs",
    required=True,
    help="how many rounds"
)

# Parse arguments
args = parser.parse_args()

# Extract values
data_dir = args.data_dir
batch_size = args.batch_size
user_history_length = args.user_history_length
user_embedding_size = args.user_embedding_size
product_embedding_size = args.product_embedding_size
hidden_size = args.hidden_size
num_epochs = args.num_epochs

# Step 1: Load the data
ratings_df = pd.read_csv(f'{data_dir}/ratings.csv')
movies_df = pd.read_csv(f'{data_dir}/movies.csv')
movies_df['movie_idx'] = movies_df.index
ratings_df = pd.merge(
    ratings_df,
    movies_df[['movieId', 'movie_idx']],
    on='movieId', how='left'
)

# Step 2: Preprocess ratings to create binary target
ratings_df['binary_rating'] = ratings_df['rating'].apply(lambda x: 1 if x >= 4 else 0)

# Step 3: Create user history (last N movies rated by the user)
print(f"user_history_length: {user_history_length}")

user_id_mapping_df = ratings_df[['userId']].drop_duplicates().reset_index(drop=True)
user_id_mapping = dict(zip(user_id_mapping_df['userId'], user_id_mapping_df.index))
ratings_df['user_idx'] = ratings_df['userId'].map(user_id_mapping)

# Create a dictionary of user histories (list of movie IDs they've rated)
user_history = {}
# Step 1: Sort the DataFrame by user and timestamp (if available)
ratings_df = ratings_df.sort_values(by=['user_idx', 'timestamp'])

# Step 2: Create a user history using groupby and apply
user_history = ratings_df.groupby('user_idx')['movie_idx'].apply(lambda x: x.values[-user_history_length:]).to_dict()

# for testing
ratings_df = ratings_df[ratings_df['user_idx'].isin(user_history.keys())].reset_index(drop=True)
ratings_df['user_idx'] = ratings_df['user_idx'].astype(int)

# Step 4: Split the data into train and test sets
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"USING | {device} for compute")

# Step 5: Prepare the dataset class
class MovieLensDataset(Dataset):
    def __init__(self, ratings_df, user_history, user_history_length=5):
        self.ratings_df = ratings_df
        self.user_history = user_history
        self.user_history_length = user_history_length
        self.users = ratings_df['user_idx'].unique()
        self.movies = ratings_df['movie_idx'].unique()

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, idx):
        user_id = self.ratings_df.iloc[idx]['user_idx']
        movie_id = self.ratings_df.iloc[idx]['movie_idx']
        binary_target = self.ratings_df.iloc[idx]['binary_rating']
        user_history = torch.tensor(self.user_history[user_id], dtype=torch.long)
        return user_id, movie_id, user_history, binary_target

# Step 6: Create DataLoader for train and test sets
full_dataset = MovieLensDataset(ratings_df, user_history, user_history_length=user_history_length)
train_dataset = MovieLensDataset(train_df, user_history, user_history_length=user_history_length)
test_dataset = MovieLensDataset(test_df, user_history, user_history_length=user_history_length)

full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=-1)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

# Step 7: Define the DIN model
class DeepInterestNetwork(nn.Module):
    def __init__(self, user_embedding_size, product_embedding_size, hidden_size, num_users, num_products):
        super(DeepInterestNetwork, self).__init__()

        # Embedding layers for users and products
        self.user_embedding = nn.Embedding(num_users, user_embedding_size)
        self.product_embedding = nn.Embedding(num_products, product_embedding_size)

        # Attention layer to compute relevance of user history
        self.attention_layer = nn.Linear(user_embedding_size, 1)

        # MLP layers for final prediction
        self.mlp1 = nn.Linear(user_embedding_size + product_embedding_size, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, 1)

    def forward(self, user_ids, product_ids, user_history):
        # Get embeddings for the user and product
        user_embed = self.user_embedding(user_ids)
        product_embed = self.product_embedding(product_ids)

        # Embed the user history
        history_embed = self.product_embedding(user_history)

        # Compute attention scores for the user history
        attention_scores = torch.tanh(self.attention_layer(history_embed))
        attention_weights = F.softmax(attention_scores, dim=1)

        # Weighted sum of user history embeddings
        weighted_history_embed = torch.sum(attention_weights * history_embed, dim=1)

        # Concatenate the weighted history and current product embedding
        x = torch.cat([weighted_history_embed, product_embed], dim=1)

        # MLP layers
        x = F.relu(self.mlp1(x))
        prediction = torch.sigmoid(self.mlp2(x))  # Output probability of positive review (0 or 1)

        return prediction



# Step 8: Model training setup
num_users = len(ratings_df['user_idx'].unique())
num_products = len(movies_df['movie_idx'].unique())

model = DeepInterestNetwork(user_embedding_size=user_embedding_size, product_embedding_size=product_embedding_size, hidden_size=hidden_size, num_users=num_users, num_products=num_products)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification

# Step 9: Training loop
print(f"RUNNING ONLY {num_epochs} EPOCHS")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print("===================")
    print(epoch+1)
    start_time = time.time()
    for user_id, movie_id, user_history, binary_target in full_dataloader:
        optimizer.zero_grad()

        # Convert inputs to LongTensor for Embedding layer
        user_id = user_id.long()
        movie_id = movie_id.long()
        user_history = user_history.long()

        # Move data to GPU
        user_id = user_id.to(device)
        movie_id = movie_id.to(device)
        user_history = user_history.to(device)
        binary_target = binary_target.to(device)


        # Forward pass
        # try:
        outputs = model(user_id, movie_id, user_history).squeeze()  # Squeeze to get rid of extra dimension
        loss = criterion(outputs, binary_target.float())  # Convert target to float for BCELoss
        loss.backward()
        # except:
        #     print((movie_id))

        
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss / len(full_dataloader)}')
    print(f'Time: {time.time() - start_time}')
    print("===================")

# Step 10: Model evaluation on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for user_id, movie_id, user_history, binary_target in full_dataloader:
        user_id = user_id.long()
        movie_id = movie_id.long()
        user_history = user_history.long()

        user_id = user_id.to(device)
        movie_id = movie_id.to(device)
        user_history = user_history.to(device)
        binary_target = binary_target.to(device)
        
        outputs = model(user_id, movie_id, user_history).squeeze()  # Get predictions
        predicted = (outputs > 0.5).float()  # Threshold at 0.5 for binary classification
        correct += (predicted == binary_target).sum().item()
        total += binary_target.size(0)

accuracy = correct / total
print(f'Train Accuracy: {accuracy * 100:.2f}%')