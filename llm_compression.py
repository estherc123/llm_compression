#model_dict = torch.load(path)
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel
import pandas as pd
import json
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import numpy as np
from scipy.linalg import svd


# Ensure that the config.json and pytorch_model.bin files are in the 'path/to/model' directory
model = AutoModel.from_pretrained('medalpaca/medalpaca-7b', torch_dtype=torch.float16, device_map = 'auto')

# Load the model weights
#model.load_state_dict(torch.load(path))

model_dict = model.state_dict()
print(model_dict.keys())


# Path to the directory with JSON files
json_files_directory = '/content/drive/My Drive/LLM_fall2023/training_sets'

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

def compute_fisher_information(model, dataloader, criterion, device='cuda'):
    model.eval()
    model.to(device)
    fisher_information = {}

    for batch in dataloader:
        inputs = batch['input'].to(device)
        targets = batch['output'].to(device)

        # Create attention mask for the inputs
        attention_mask = (inputs != 0).long()

        model.zero_grad()

        outputs = model(inputs, attention_mask=attention_mask)
        # Access the logits from the model's output
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        # Reshape logits and targets if necessary
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        loss = criterion(logits, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if 'weight' in name and not ('layernorm' in name):
                if param.grad is not None:
                    fisher_information[name] = fisher_information.get(name, 0) + param.grad.pow(2).sum().item()
    return fisher_information



class CustomDataset(Dataset):
    def __init__(self, json_data, tokenizer, max_length=512):
        self.max_length = max_length
        self.data = json_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        # If json_data is a string representing a list of dictionaries
        self.data = json_data
        self.data = self.data.reset_index(drop=True)  # Reset index if it's a pandas DataFrame

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row by index
        sample = self.data.iloc[idx]
        # Process the 'input' and 'output' text data
        #instruction = sample['instruction']
        input_text = sample['input']
        output_text = sample['output']

        # Handle tokenization and conversion to tensors

        input_tokens =  self.tokenizer.encode_plus(
    input_text,
    add_special_tokens=True,
    max_length=512,  # Set the maximum length
    truncation=True,  # Enable truncation to max_length
    padding='max_length'  # Pad shorter sequences
)
        output_tokens = self.tokenizer.encode_plus(
    output_text,
    add_special_tokens=True,
    max_length=512,  # Set the maximum length
    truncation=True,  # Enable truncation to max_length
    padding='max_length'  # Pad shorter sequences
)


        return {
            #'instruction': instruction_tensor,
            'input': input_tokens,
            'output': output_tokens,
            'length': len(input_tokens)  # Add the length for padding purposes
        }



def collate_fn(batch):
    inputs = [item['input'] for item in batch]
    outputs = [item['output'] for item in batch]

    print(type(batch[0]['input']), batch[0]['input'])
    print(type(batch[0]['output']), batch[0]['output'])
    inputs_padded = pad_sequence([torch.tensor(seq['input_ids']) for seq in inputs], batch_first=True, padding_value=0)
    outputs_padded = pad_sequence([torch.tensor(seq['input_ids']) for seq in outputs], batch_first=True, padding_value=0)

    lengths = torch.tensor([len(seq) for seq in inputs])

    return {
        'input': inputs_padded,
        'output': outputs_padded,
        'length': lengths
    }



# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Initialize the custom dataset with your data and the tokenizer
dataset = CustomDataset(sampled_data, tokenizer, max_length = 512)
# When initializing the DataLoader, pass the custom collate function
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)



# Assuming model outputs logits of shape (batch_size, sequence_length, vocab_size)
# and targets of shape (batch_size, sequence_length)
def compute_loss(model_outputs, targets):
    # Define the criterion
    criterion = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Compute the loss
    loss = criterion(model_outputs, targets)

    return loss

fisher_information = {}

# Commented out IPython magic to ensure Python compatibility.
# %env CUDA_LAUNCH_BLOCKING=

compute_fisher_information(model, dataloader, compute_loss, device='cuda')



fw_weights = {}
fisher_information = {}

def compute_weighted_matrix(param_name, weight_matrix):
    return fisher_information[param_name] * weight_matrix

def fw_svd(param_name, weighted_matrix, r):
    # SVD on the weighted matrix
    U, S, V = np.linalg.svd(compute_weighted_matrix(param_name, weighted_matrix), full_matrices=False)

    # Truncate to retain only r ranks
    U_r, S_r, V_r = U[:, :r], np.diag(S[:r]), V[:r, :]

    # Compute I_hat
    I_hat = np.sqrt(np.sum(weighted_matrix**2, axis=1))
    I_hat_inv = np.diag(1.0 / I_hat)

    # Compute A and B
    A = np.dot(I_hat_inv, U_r).dot(S_r)
    B = V_r.T

    return np.matmul(A,B)

def compress(param_name, weight_matrix, r):

    # SVD on the weighted matrix
    U, S, V = np.linalg.svd(weight_matrix, full_matrices=False)

    # Truncate to retain only r ranks
    U_r, S_r, V_r = U[:, :r], np.diag(S[:r]), V[:r, :]

    # Compute A and B
    A = np.dot(U_r, np.dot(S_r, V_r))


    return A



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
  print(key)
  if 'weight' in key and 'layernorm' not in key:
    # Assume weight_matrix is your layer's weight matrix
    weight_matrix = model_dict[key]

    # Choose a variance threshold
    variance_threshold = 0.95  # for example, to retain 95% of variance

    # Calculate the optimal k
    #optimal_k, compressed_W = find_optimal_k(weight_matrix, variance_threshold)
    model_dict[key] = fw_swd(key, weight_matrix, optimal_k)




torch.save(model_dict, '/results/compressed_model_state_dict.bin')
