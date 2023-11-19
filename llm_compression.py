import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import os
import json
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from scipy.linalg import svd
# Ensure that the config.json and pytorch_model.bin files are in the 'path/to/model' directory
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')

# Path to the directory with JSON files
json_files_directory = '/path/to/training/set'

# List all JSON files in the directory
json_files = [pos_json for pos_json in os.listdir(json_files_directory) if pos_json.endswith('.json')]

# Combine all JSON files into a single DataFrame
combined_data = pd.DataFrame()
for file_name in json_files:
    file_path = os.path.join(json_files_directory, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)
        df = pd.DataFrame(data)  # or pd.read_json(file_path) if each file is a JSON array
        combined_data = combined_data.append(df, ignore_index=True)

# Shuffle the combined data
shuffled_data = shuffle(combined_data)

# Sample a portion of the data, let's say 1%
sampled_data = shuffled_data.sample(frac=0.01)

def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = pad_token_id

    assert pad_token_id is not None, "pad_token_id has to be defined."
    # Replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids



class CustomDataset(Dataset):
    def __init__(self, json_data, tokenizer, max_length=512):
        self.data = json_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        input_text = sample['input']
        output_text = sample['output']

        input_tokens = self.tokenizer(input_text, add_special_tokens=True, max_length=self.max_length, truncation=True, padding='max_length')
        output_tokens = self.tokenizer(output_text, add_special_tokens=True, max_length=self.max_length, truncation=True, padding='max_length')

        # Convert to tensors and ensure float16
        input_ids = torch.tensor(input_tokens['input_ids'], dtype=torch.long)
        output_ids = torch.tensor(output_tokens['input_ids'], dtype=torch.long)

        return {'input': input_ids, 'output': output_ids}



def collate_fn(batch):
    inputs = [item['input'] for item in batch]
    outputs = [item['output'] for item in batch]

    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    outputs_padded = pad_sequence(outputs, batch_first=True, padding_value=0)
    #print("padded inputs: ", inputs_padded.shape)
    #print("padded outputs: ", outputs_padded.shape)
    return {'input': inputs_padded, 'output': outputs_padded}


# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')

# Initialize the custom dataset with your data and the tokenizer
dataset = CustomDataset(sampled_data, tokenizer, max_length = 512)
# When initializing the DataLoader, pass the custom collate function
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

def compute_fisher_information(model, dataloader, criterion, device='cpu'):
    model.eval()
    model.to(device)
    counter = 0  # Initialize a counter
    total_batches = len(dataloader)  # Total number of batches
    for batch in dataloader:
        print(f"Processing batch {counter}/{total_batches}")
        counter += 1  # Increment the counter
        inputs = batch['input'].to(device)
        targets = batch['output'].to(device)
        batch_size = inputs.size(0)
        sequence_length = inputs.size(1)
        # Create attention mask for the inputs
        attention_mask = (inputs != 0).long()
                # Shift output_ids to the right to create decoder_input_ids
        decoder_input_ids = shift_tokens_right(targets, model.config.pad_token_id)

        model.zero_grad()
        #print("inputs shape: ", inputs.shape, "targets shape: ", targets.shape)
        outputs = model(input_ids=inputs, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        vocab_size = tokenizer.vocab_size+28
        #print("vocab size: ", vocab_size)
        #print("logits: ", logits.shape)
        #print("targets: ", targets.shape)
        # Reshape logits and targets if necessary
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)


        # print("vocab size: ", vocab_size)
        #vocab_size = (int)(8192/batch_size)
        logits = logits.view(batch_size, sequence_length, vocab_size)
        targets = targets.view(batch_size, sequence_length)

        if (targets >= vocab_size).any():
            print("Invalid target values found.")

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("NaN or Inf in logits")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("NaN or Inf in targets")

        loss = criterion(logits, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if 'weight' in name and not ('layernorm' in name):
                if param.grad is not None:
                    fisher_information[name] = fisher_information.get(name, 0) + param.grad.pow(2).sum().item()
    return fisher_information



# Assuming model outputs logits of shape (batch_size, sequence_length, vocab_size)
# and targets of shape (batch_size, sequence_length)
def compute_loss(model_outputs, targets):
    # Define the criterion
    criterion = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    # Compute the loss
    loss = criterion(model_outputs.transpose(1, 2), targets)

    return loss

fisher_information = {}
compute_fisher_information(model, dataloader, compute_loss, device='cuda')

fw_weights = {}
#fisher_information = {}

def compute_weighted_matrix(param_name, weight_matrix):
    return fisher_information[param_name] * weight_matrix

def fw_svd(param_name, weighted_matrix, r):
    # SVD on the weighted matrix
    #print(weighted_matrix.shape)
    U, S, V = np.linalg.svd(compute_weighted_matrix(param_name, weighted_matrix), full_matrices=False)

    # Truncate to retain only r ranks
    U_r, S_r, V_r = U[:, :r], np.diag(S[:r]), V[:r, :]
    #print(U_r.shape, U.shape)
    # Compute I_hat
    I_hat = np.sqrt(np.sum(weighted_matrix**2, axis=1))
    I_hat_inv = np.diag(1.0 / I_hat)

    # Compute A and B
    A = np.dot(I_hat_inv, U_r).dot(S_r)
    B = V_r

    #print(A.shape)
    #print(B.shape)
    return np.matmul(A,B)

model_dict = model.state_dict()

def find_optimal_k(weight_matrix, variance_threshold=0.95):
    # Perform Singular Value Decomposition
    U, S, Vt = svd(weight_matrix, full_matrices=False)

    # Calculate the explained variance ratios for each singular value
    explained_variances = np.square(S) / np.sum(np.square(S))

    # Calculate the cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variances)

    # Find the number of singular values needed to retain the specified threshold of variance
    k = np.argmax(cumulative_explained_variance >= variance_threshold) + 1

    return k, cumulative_explained_variance

for key in model_dict.keys():
  if 'weight' in key and ('layer_norm' not in key) and ('embed_tokens.weight' not in key):
    # Assume weight_matrix is your layer's weight matrix
    print(key)
    weight_matrix = model_dict[key].cpu().detach().numpy() if torch.is_tensor(model_dict[key]) else model_dict[key]
    #print(weight_matrix)
    # Choose a variance threshold
    variance_threshold = 0.95  # for example, to retain 95% of variance

    # Calculate the optimal k
    optimal_k, compressed_W = find_optimal_k(weight_matrix, variance_threshold)
    model_dict[key] = fw_svd(key, weight_matrix,optimal_k)


torch.save(model_dict, '/destination/path')
