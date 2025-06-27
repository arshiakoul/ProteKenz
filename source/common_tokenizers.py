import time
from sklearn.metrics import matthews_corrcoef, accuracy_score
from scipy import stats
import torch
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
                                UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
import os
import logging
from torch.optim import Adam
from torch import nn
from transformers import BertModel, BertConfig
import torch
import json
import os
import json
from pyrosetta import *
import json
from pathlib import Path
from typing import List

# reference: https://github.com/idotan286/BiologicalTokenizers/blob/main/train_tokenizer_bert.py#L208

###########################################
# Configuration and Initialization
###########################################
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

DF_NAME = 'sampled_pdb.tsv'
TRAIN_DF_NAME = "train.csv"
VALID_DF_NAME = "valid.csv"
TEST_DF_NAME = "test.csv"

TASK_REGRESSION = "REGRESSION"
TASK_CLASSIFICATION = "CLASSIFICATION"
TASK_CATEGORICAL = "CATEGORICAL"

TOKEZNIER_BPE = "BPE"
TOKEZNIER_UNI = "UNI"
TOKEZNIER_AA = "AA"
TOKENIZER_CUSTOM_SIMPLE = "CUSTOM_SIMPLE"

UNK_TOKEN = "<UNK>"  # token for unknown AA
PAD_TOKEN = "<PAD>"  # token for padding
CLS_TOKEN = "<CLS>"  # token for classification
MASK_TOKEN = "<MASK>"  # token for masking
SEP_TOKEN = "<SEP>"  # token for separation
SPL_TOKENS = [UNK_TOKEN, PAD_TOKEN, CLS_TOKEN, MASK_TOKEN, SEP_TOKEN]  # list of special tokens

def generate_top_k_vocab(vocab_file: str, k_sizes: List[int], save_dir: str):
    """
    Generates multiple vocabulary lists with different top-k token selections.
    
    :param vocab_file: Path to the original vocabulary file (TXT format).
    :param k_sizes: List of vocabulary sizes to generate (e.g., [50, 100, 500]).
    :param save_dir: Directory to save the generated vocabularies.
    """
    vocab_path = Path(vocab_file)
    if not vocab_path.exists():
        raise ValueError(f"Vocabulary file not found at {vocab_file}")

    # Read and extract token-frequency pairs
    token_freqs = {}
    with open(vocab_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(":")
            if len(parts) == 2:
                token, freq = parts[0].strip(), float(parts[1].strip())
                token_freqs[token] = freq

    # Sort tokens by frequency in descending order
    sorted_tokens = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)

    # Create the save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    # Special tokens
    special_tokens = {
        "<UNK>": 0,
        "<PAD>": 1,
        "<CLS>": 2,
        "<MASK>": 3,
        "<SEP>": 4
    }

    # Generate vocabularies for each requested size
    for k in k_sizes:
        top_k_tokens = sorted_tokens[:k-len(special_tokens)]  # Exclude special tokens
        vocab_dict = {token: idx for idx, (token, _) in enumerate(top_k_tokens)}
        # Save the vocabulary as a JSON file
        vocab_filename = f"vocab_{k}.json"
        vocab_path = Path(save_dir) / vocab_filename
        with open(vocab_path, "w") as f:
            json.dump(vocab_dict, f, indent=2)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, attention_masks, labels):
        self.encodings = encodings
        self.attention_masks = attention_masks
        self.labels = labels

    def __getitem__(self, idx):
        return self.encodings[idx], self.attention_masks[idx], self.labels[idx]

    def __len__(self):
        return len(self.encodings)

def prepare_tokenizer_trainer(alg, voc_size):
    """
    Prepares the tokenizer and trainer with unknown & special tokens.
    """
    if alg == TOKEZNIER_BPE:
        tokenizer = Tokenizer(BPE(unk_token = UNK_TOKEN))
        trainer = BpeTrainer(special_tokens = SPL_TOKENS, vocab_size=voc_size)
    elif alg == TOKEZNIER_UNI:
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token= UNK_TOKEN, special_tokens = SPL_TOKENS, vocab_size=voc_size)
    elif alg == TOKEZNIER_AA:
        tokenizer = Tokenizer(WordLevel(unk_token = UNK_TOKEN))
        trainer = WordLevelTrainer(special_tokens = SPL_TOKENS)
    elif alg == TOKENIZER_CUSTOM_SIMPLE:
        tokenizer = Tokenizer(WordLevel(unk_token = UNK_TOKEN))
        trainer = WordLevelTrainer(special_tokens = SPL_TOKENS)
    else:
        exit(f'unknown tokenizer type, please use one of the following: ["{TOKEZNIER_BPE}", "{TOKEZNIER_UNI}", "{TOKEZNIER_AA}"]')
    
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer

def train_tokenizer(iterator, alg, vocab_size):
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer(alg, vocab_size)
    tokenizer.train_from_iterator(iterator, trainer) # training the tokenzier
    return tokenizer

def batch_iterator(dataset):
    batch_size = 10000
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def greedy_tokenization(sequence, token_set):
        """
        Tokenizes a protein sequence using a predefined set of tokens.
        Uses a greedy approach to find the longest possible match at each step.

        :param sequence: The protein sequence (string) to tokenize.
        :param token_set: A set of known tokens (for fast lookup).
        :return: List of tokens representing the tokenized sequence.
        """
        max_token_length = max(map(len, token_set))  # Find the longest token length
        tokenized_sequence = []
        i = 0  # Pointer for the sequence

        while i < len(sequence):
            matched = False  # Track if a token was matched

            # Try the longest possible token first
            for length in range(min(max_token_length, len(sequence) - i), 0, -1):
                token = sequence[i:i + length]
                if token in token_set:
                    tokenized_sequence.append(token)
                    i += length  # Move forward by the token length
                    matched = True
                    break  # Stop checking once the longest token is found

            # If no token matches, use single amino acid as a fallback
            if not matched:
                tokenized_sequence.append(sequence[i])
                i += 1

        return tokenized_sequence

def train_biological_tokenizer(data_path, tokenizer_type, vocab_size, results_path, max_length=512):
    df = pd.read_csv(os.path.join(data_path, DF_NAME), sep='\t')
    X = df['Sequence'].astype(str).tolist()

    if 'AA' == tokenizer_type:
        #single_amino_acids = ['A', 'I', 'V', 'G', 'D', 'T', 'Z', 'Y', 'W', 'K', 'H', 'O', 'S', 'C', 'B', 'U', 'E', 'N', 'M', 'Q', 'F', 'L', 'X', 'R', 'P']
        X = [' '.join([*aminos]) for aminos in X]
    
    if 'CUSTOM_SIMPLE' == tokenizer_type:
        vocab = f'results/simple_tokenization/vocabularies/new/vocab_{vocab_size}.json'
        with open(vocab, "r") as f:
            vocab_dict = json.load(f)
        vocabs = list(vocab_dict.keys())
        X = [' '.join(greedy_tokenization(X[i], vocabs)) for i in range(len(X))]
    
    logger.info(f'starting to train {tokenizer_type} tokenizer...')
    tokenizer = train_tokenizer(batch_iterator(X), tokenizer_type, vocab_size)
    tokenizer.enable_padding(length=max_length)
    logger.info(f'saving tokenizer to {results_path}...')
    if tokenizer_type == 'AA':
        tokenizer.save(os.path.join(results_path, f'{tokenizer_type}_sampled_tokenizer.json'))
    else:
        tokenizer.save(os.path.join(results_path, f'{tokenizer_type}_{vocab_size}_sampled_tokenizer.json'))

def use_tokenizer(data_path, task_type, tokenizer_type, tokenizer, max_length):
    """
    Reads the data from folder, trains the tokenizer, encode the sequences and returns list of data for BERT training
    """
    df_train = pd.read_csv(os.path.join(data_path, TRAIN_DF_NAME))
    df_valid = pd.read_csv(os.path.join(data_path, VALID_DF_NAME))
    df_test = pd.read_csv(os.path.join(data_path, TEST_DF_NAME))

    if task_type == TASK_REGRESSION:
        logger.info(f'starting a REGRESSION task!')
        y_train = df_train['label'].astype(float).tolist()
        y_valid = df_valid['label'].astype(float).tolist()
        y_test = df_test['label'].astype(float).tolist()
        num_of_classes = 1
    elif task_type == TASK_CLASSIFICATION:
        logger.info(f'starting a CLASSIFICATION task!')
        df_train['label_numeric'] = pd.factorize(df_train['label'], sort=True)[0]
        df_valid['label_numeric'] = pd.factorize(df_valid['label'], sort=True)[0]
        df_test['label_numeric'] = pd.factorize(df_test['label'], sort=True)[0]
        y_train = df_train['label_numeric'].astype(int).tolist()
        y_valid = df_valid['label_numeric'].astype(int).tolist()
        y_test = df_test['label_numeric'].astype(int).tolist()
        num_of_classes = len(list(set(y_train))) # counts the number different classes
    elif task_type == TASK_CATEGORICAL:
        logger.info(f'starting a CATEGORICAL task!')
        # Convert sequence labels into lists of integers
        y_train = [list(map(int, list(label))) for label in df_train['label']]
        y_valid = [list(map(int, list(label))) for label in df_valid['label']]
        y_test = [list(map(int, list(label))) for label in df_test['label']]
        
        # Find number of unique labels
        unique_labels = sorted(set.union(*df_train['label'].apply(set)) | set.union(*df_valid['label'].apply(set)) | set.union(*df_test['label'].apply(set)))
        num_of_classes = len(unique_labels)
    else:
        exit(f'unknown type of task, got {task_type}. Aviable options are: {TASK_REGRESSION} for regression or {TASK_CLASSIFICATION} for classification')
    
    X_train = df_train['seq'].astype(str).tolist()
    X_valid = df_valid['seq'].astype(str).tolist()
    X_test = df_test['seq'].astype(str).tolist()

    if 'AA' == tokenizer_type:
        X_train = [' '.join([*aminos]) for aminos in X_train]
        X_valid = [' '.join([*aminos]) for aminos in X_valid]
        X_test = [' '.join([*aminos]) for aminos in X_test]
    if 'CUSTOM_SIMPLE' == tokenizer_type:
        vocab = tokenizer.get_vocab()
        vocab_list = list(vocab.keys())
        X_train =  [' '.join(greedy_tokenization(X_train[i], vocab_list)) for i in range(len(X_train))]
        X_valid = [' '.join(greedy_tokenization(X_valid[i], vocab_list)) for i in range(len(X_valid))]
        X_test = [' '.join(greedy_tokenization(X_test[i], vocab_list)) for i in range(len(X_test))]

    def encode_categorical(X, Y):
        result = []
        masks = []
        label_ids = []
        
        for x, y in zip(X, Y):
            encoding = tokenizer.encode(x)
            ids = encoding.ids
            attn_mask = [1] * len(ids)

            if len(ids) > max_length:
                ids = ids[:max_length]
                attn_mask = attn_mask[:max_length]
                y = y[:max_length]  # Trim labels as well
            else:
                pad_length = max_length - len(ids)
                pad_id = tokenizer.token_to_id(PAD_TOKEN)  # Pad ID is 1
                ids += [pad_id] * pad_length
                attn_mask += [0] * pad_length
                y += [-1] * pad_length  # Pad labels with 1 (ignored during loss computation)

            result.append(ids)
            masks.append(attn_mask)
            label_ids.append(y)

        return result, masks, label_ids
    
    def encode(X):            
        result = []
        masks = []
        for x in X:
            encoding = tokenizer.encode(x)
            ids = encoding.ids
            attn_mask = [1] * len(ids)

            if len(ids) > max_length:
                ids = ids[:max_length]
                attn_mask = attn_mask[:max_length]
            else:
                pad_length = max_length - len(ids)
                ids += [tokenizer.token_to_id(PAD_TOKEN)] * pad_length
                attn_mask += [0] * pad_length
            result.append(ids)
            masks.append(attn_mask)

        return result, masks

    if task_type == TASK_CATEGORICAL:
        X_train_ids, X_train_masks, y_train_ids = encode_categorical(X_train, y_train)
        X_valid_ids, X_valid_masks, y_valid_ids = encode_categorical(X_valid, y_valid)
        X_test_ids, X_test_masks, y_test_ids = encode_categorical(X_test, y_test)

        X_train_ids = [torch.tensor(item).to(device) for item in X_train_ids]
        X_train_masks = [torch.tensor(item).to(device) for item in X_train_masks]
        y_train_ids = [torch.tensor(item).to(device) for item in y_train_ids]

        X_valid_ids = [torch.tensor(item).to(device) for item in X_valid_ids]
        X_valid_masks = [torch.tensor(item).to(device) for item in X_valid_masks]
        y_valid_ids = [torch.tensor(item).to(device) for item in y_valid_ids]

        X_test_ids = [torch.tensor(item).to(device) for item in X_test_ids]
        X_test_masks = [torch.tensor(item).to(device) for item in X_test_masks]
        y_test_ids = [torch.tensor(item).to(device) for item in y_test_ids]

        train_dataset = Dataset(X_train_ids, X_train_masks, y_train_ids)
        valid_dataset = Dataset(X_valid_ids, X_valid_masks, y_valid_ids)
        test_dataset = Dataset(X_test_ids, X_test_masks, y_test_ids)
    else:
        X_train_ids, X_train_masks = encode(X_train)
        X_valid_ids, X_valid_masks = encode(X_valid)
        X_test_ids, X_test_masks = encode(X_test)
        
        X_train_ids = [torch.tensor(item).to(device) for item in X_train_ids]
        X_train_masks = [torch.tensor(item).to(device) for item in X_train_masks]
        y_train = [torch.tensor(item).to(device) for item in y_train]

        X_valid_ids = [torch.tensor(item).to(device) for item in X_valid_ids]
        X_valid_masks = [torch.tensor(item).to(device) for item in X_valid_masks]
        y_valid = [torch.tensor(item).to(device) for item in y_valid]

        X_test_ids = [torch.tensor(item).to(device) for item in X_test_ids]
        X_test_masks = [torch.tensor(item).to(device) for item in X_test_masks]
        y_test = [torch.tensor(item).to(device) for item in y_test]

        train_dataset = Dataset(X_train_ids, X_train_masks, y_train)
        valid_dataset = Dataset(X_valid_ids, X_valid_masks, y_valid)
        test_dataset = Dataset(X_test_ids, X_test_masks, y_test)

    return num_of_classes, train_dataset, valid_dataset, test_dataset

class BioBERTModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_attention_heads, num_classes):
        super(BioBERTModel, self).__init__()
        configuration = BertConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, num_attention_heads=num_attention_heads)
        self.transformer = BertModel(configuration)
        
        # additional layers for the classification / regression task
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
       )

    def forward(self, ids, mask=None):
        sequence_output, pooled_output = self.transformer(
            ids, 
            attention_mask=mask,
            return_dict=False
        )
        sequence_output = torch.mean(sequence_output, dim=1)
        return self.head(sequence_output)
    
class BioBERTModelCategorical(nn.Module):
    def __init__(self, hidden_size, num_layers, num_attention_heads, num_classes):
        super(BioBERTModelCategorical, self).__init__()
        configuration = BertConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, num_attention_heads=num_attention_heads)
        self.transformer = BertModel(configuration)
        
        # Token-wise classification layer
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=-1)  # Apply softmax per token
        )

    def forward(self, ids, mask=None):
        sequence_output, pooled_output = self.transformer(
            ids, 
            attention_mask=mask,
            return_dict=False
        )
        return self.head(sequence_output)  # Output shape: (batch_size, seq_length, num_classes)

def train_model(model, task_type, train_generator, valid_generator, test_generator, epochs, print_training_logs, results_path):
    os.makedirs(results_path, exist_ok=True)
    if task_type == TASK_REGRESSION:
        loss_fn = nn.MSELoss()
    elif task_type == TASK_CATEGORICAL:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.00002)
    
    def calc_metrics_regression(model, generator):
        loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x,mask,y in generator:
                outputs = model(x, mask)  
                outputs = outputs.to(torch.float)
                y_pred.append(outputs[0].item())
                y = y.to(torch.float)
                loss += loss_fn(outputs, y).item()
                y_true.append(y.item())
            loss = loss / len(generator)
            spearman = stats.spearmanr(y_pred, y_true)
        return loss, spearman[0], spearman[1]
    
    def calc_metrics_classification(model, generator):
        loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x,mask,y in generator:
                outputs = model(x, mask)  
                y_pred.append(torch.argmax(outputs, dim=1).int().item())
                loss += loss_fn(outputs, y).item()
                y_true.append(y.int().item())
            loss = loss / len(generator)
            mcc = matthews_corrcoef(y_true, y_pred)
            accuracy = accuracy_score(y_true, y_pred)
        return loss, mcc, accuracy
    
    def calc_metrics_categorical(model, generator):
        import numpy as np
        """
        Evaluates model performance for per-token classification, ignoring PAD tokens.
        """
        loss = 0
        y_true_all = []
        y_pred_all = []
        
        with torch.no_grad():
            for x, mask, y in generator:
                outputs = model(x, mask)  # Shape: (batch_size, seq_length, num_classes)
                
                # Flatten predictions and labels
                y_pred = torch.argmax(outputs, dim=-1)  # Get class predictions
                y_pred = y_pred.view(-1)  # Flatten: (batch_size * seq_length)
                y_true = y.view(-1)  # Flatten: (batch_size * seq_length)
                
                # Apply loss, ignoring PAD tokens (ID=1)
                loss += loss_fn(outputs.view(-1, outputs.shape[-1]), y_true).item()

                # Mask out PAD tokens from evaluation
                valid_idx = y_true != -1  # Mask where y_true != PAD_ID (1)
                y_true = y_true[valid_idx]
                y_pred = y_pred[valid_idx]
                
                y_true_all.append(y_true.cpu().numpy())
                y_pred_all.append(y_pred.cpu().numpy())

        # Compute metrics
        y_true_all = np.concatenate(y_true_all, axis=0)
        y_pred_all = np.concatenate(y_pred_all, axis=0)
        
        accuracy = accuracy_score(y_true_all, y_pred_all)
        
        return loss / len(generator), accuracy
        
    list_of_rows = []
    if task_type == TASK_CATEGORICAL:
        for epoch in range(1, epochs + 1):
            logger.info(f'----- starting epoch = {epoch} -----')
            running_loss = 0.0
            start_time = time.time()
            model.train()

            for idx, (x, mask, y) in enumerate(train_generator):
                optimizer.zero_grad()
                outputs = model(x, mask)  # Shape: (batch_size, seq_length, num_classes)

                # Compute loss
                loss = loss_fn(outputs.view(-1, outputs.shape[-1]), y.view(-1))  # Flatten for loss computation
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if idx % print_training_logs == print_training_logs - 1:
                    end_time = time.time()
                    logger.info('[%d, %5d] time: %.3f loss: %.3f' %
                        (epoch, idx + 1, end_time - start_time, running_loss / print_training_logs))
                    running_loss = 0.0
                    start_time = time.time()
            
            model.eval()
            val_loss, accuracy_val = calc_metrics_categorical(model, valid_generator)
            test_loss, accuracy_test = calc_metrics_categorical(model, test_generator)
            
            logger.info(f'epoch = {epoch}, val_loss = {val_loss}, accuracy_val = {accuracy_val}, test_loss = {test_loss}, accuracy_test = {accuracy_test}')
            list_of_rows.append({'epoch': epoch, 'val_loss': val_loss, 'accuracy_val': accuracy_val, 'test_loss': test_loss, 'accuracy_test': accuracy_test})
            torch.save(model.state_dict(), os.path.join(results_path, f"checkpoint_{epoch}.pt"))
    else:
        for epoch in range(1, epochs + 1):
            logger.info(f'----- starting epoch = {epoch} -----')
            running_loss = 0.0
            # Training
            start_time = time.time()
            model.train()
            for idx, (x, mask, y) in enumerate(train_generator):
                optimizer.zero_grad()
                outputs = model(x, mask)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()
                
                # print statistics
                running_loss += loss.item()
                if idx % print_training_logs == print_training_logs - 1:
                    end_time = time.time()
                    logger.info('[%d, %5d] time: %.3f loss: %.3f' %
                        (epoch, idx + 1, end_time - start_time, running_loss / print_training_logs))
                    running_loss = 0.0
                    start_time = time.time()
            
            model.eval()
            if task_type == TASK_REGRESSION:
                val_loss, spearman_val_corr, spearman_val_p = calc_metrics_regression(model, valid_generator)
                test_loss, spearman_test_corr, spearman_test_p = calc_metrics_regression(model, test_generator)
                
                logger.info(f'epoch = {epoch}, val_loss = {val_loss}, spearman_val_corr = {spearman_val_corr}, spearman_val_p = {spearman_val_p}, test_loss = {test_loss}, spearman_test_corr = {spearman_test_corr}, spearman_test_p = {spearman_test_p}')
                list_of_rows.append({'epoch': epoch, 'val_loss': val_loss, 'spearman_val_corr': spearman_val_corr, 'spearman_val_p': spearman_val_p, 'test_loss': test_loss, 'spearman_test_corr': spearman_test_corr, 'spearman_test_p': spearman_test_p})
            
            else:
                val_loss, val_mcc, accuracy_val = calc_metrics_classification(model, valid_generator)
                test_loss, test_mcc, accuracy_test = calc_metrics_classification(model, test_generator)
                
                logger.info(f'epoch = {epoch}, val_loss = {val_loss}, val_mcc = {val_mcc}, test_loss = {test_loss}, test_mcc = {test_mcc}, accuracy_val = {accuracy_val}, accuracy_test = {accuracy_test}')
                list_of_rows.append({'epoch': epoch, 'val_loss': val_loss, 'val_mcc': val_mcc, 'test_loss': test_loss, 'test_mcc': test_mcc, 'accuracy_val': accuracy_val,'accuracy_test': accuracy_test})
            
            torch.save(model.state_dict(), os.path.join(results_path, f"checkpoint_{epoch}.pt"))
        
    df_loss = pd.DataFrame(list_of_rows)
    df_loss.to_csv(os.path.join(results_path, f"results.csv"), index=False)


if __name__ == "__main__":
    results_path = 'results/simple_tokenization'
    vocab_size = 200
    tokenizer_type = 'CUSTOM_SIMPLE'
    class_type = 'CLASSIFICATION'
    bench_path = 'data/benchmark1/SuperFamily'
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(results_path, exist_ok=True)
    vocab_file_path = "results/simple_tokenization/vocabulary_epoch_3_seq_700_new.txt"  # Path to the original vocab file
    save_directory = "results/simple_tokenization/vocabularies/new"  # Directory to save generated vocabularies
    k_values = [50, 100, 200, 400, 800, 1600, 3200]  # Different vocab sizes to generate

    generate_top_k_vocab(vocab_file_path, k_values, save_directory)
    
    if tokenizer_type == 'CUSTOM_SIMPLE':
        if os.path.exists(f'{results_path}/vocabularies/new/{tokenizer_type}_{vocab_size}_sampled_tokenizer.json') == False:
            train_biological_tokenizer("data", tokenizer_type, vocab_size, f"{results_path}/vocabularies/new")
        tokenizer = Tokenizer.from_file(f'{results_path}/vocabularies/new/{tokenizer_type}_{vocab_size}_sampled_tokenizer.json')
    elif tokenizer_type == 'AA':
        if os.path.exists(f'{results_path}/{tokenizer_type}_sampled_tokenizer.json') == False:
            train_biological_tokenizer("data", tokenizer_type, vocab_size, results_path)
        tokenizer = Tokenizer.from_file(f'{results_path}/{tokenizer_type}_sampled_tokenizer.json')
    else:
        if os.path.exists(f'{results_path}/{tokenizer_type}_{vocab_size}_sampled_tokenizer.json') == False:
             train_biological_tokenizer("data", tokenizer_type, vocab_size, results_path)
        tokenizer = Tokenizer.from_file(f'{results_path}/{tokenizer_type}_{vocab_size}_sampled_tokenizer.json')
            
    num_classes, train_dataset, valid_dataset, test_dataset = use_tokenizer(bench_path, class_type, tokenizer_type, tokenizer, 512)
    
    if class_type == 'CATEGORICAL':
        model = BioBERTModelCategorical(128, 2, 2, num_classes)
    else:
        model = BioBERTModel(128, 2, 2, num_classes)
    model.to(device)
    logger.info(f'loaded model to device')
    logger.info(f'device is {device}')
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'num of paramters = {total_params}')
    
    g = torch.Generator()
    g.manual_seed(0)
    train_generator = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=8, generator=g)
    valid_generator = torch.utils.data.DataLoader(valid_dataset, shuffle=True, num_workers=0, batch_size=1, generator=g)
    test_generator = torch.utils.data.DataLoader(test_dataset, shuffle=True, num_workers=0, batch_size=1, generator=g)
    if tokenizer_type == 'UNI' or tokenizer_type == 'BPE':
        train_model(model, class_type, train_generator, valid_generator, test_generator, 30, 1000, f'{results_path}/{tokenizer_type}/{vocab_size}')
    elif tokenizer_type == 'AA':
        train_model(model, class_type, train_generator, valid_generator, test_generator, 30, 1000, f'{results_path}/{tokenizer_type}')
    else:
        train_model(model, class_type, train_generator, valid_generator, test_generator, 30, 1000, f'{results_path}/{vocab_size}')
