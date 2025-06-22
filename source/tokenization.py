import os
import random
import numpy as np
import json
from collections import OrderedDict
from typing import List, Tuple
import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.core.scoring import fa_atr, fa_rep, fa_sol, EMapVector
import matplotlib.pyplot as plt
import csv
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

from transformers import PreTrainedTokenizer
###########################################
# Configuration and Initialization
###########################################

class Config:
    def __init__(self,
                 token_sizes: List[int] = [random.randint(3, 10) for _ in range(5)],
                 gamma: float = 0.8,
                 alpha: float = 0.1,
                 epsilon: float = 0.1,
                 num_episodes: int = 10,
                 reward_scale: float = 1.0,
                 max_q_table_size: int = 1000):
        self.token_sizes = token_sizes
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.reward_scale = reward_scale
        self.max_q_table_size = max_q_table_size
###########################################
# Utility Functions
###########################################
def read_pdb_files_from_directory(directory: str) -> List[Tuple[str, str, str]]:
    """
    Reads PDB files from a directory and extracts:
      - A uniprot id (assumed as the second part when splitting the filename by "-"),
      - The sequence (using PyRosetta's pose.sequence()),
      - And the pdb file path.
      
    To speed up subsequent runs, the results are cached in 'pdb_cache.json'
    within the same directory. If the cache file exists, it is loaded instead of
    reprocessing all PDB files.
    """
    cache_file = os.path.join(directory, "pdb_cache.json")
    if os.path.exists(cache_file):
        print("Loading PDB data from cache...")
        with open(cache_file, 'r') as f:
            # The cache stores a list of lists; each inner list contains [uniprot_id, sequence, pdb_path]
            sequences_and_pdbs = json.load(f)
        return sequences_and_pdbs
    else:
        print("Cache not found. Reading PDB files (this may take some time)...")
        pdb_files = [f for f in os.listdir(directory) if f.endswith(".pdb")]
        sequences_and_pdbs = []
        for pdb_file in pdb_files:
            pdb_path = os.path.join(directory, pdb_file)
            pose = pose_from_pdb(pdb_path)
            sequence = pose.sequence()
            parts = pdb_file.split("-")
            uniprot_id = parts[1] if len(parts) > 1 else pdb_file
            sequences_and_pdbs.append([uniprot_id, sequence, pdb_path])
        with open(cache_file, 'w') as f:
            json.dump(sequences_and_pdbs, f)
        print(f"Cached PDB data to {cache_file}")
        return sequences_and_pdbs

def read_test_dataset(csv_file: str) -> List[Tuple[str, str]]:
    """
    Reads a test dataset CSV file with columns 'ec_number' and 'sequence'.
    Returns a list of tuples: (ec_number, sequence).
    """
    test_data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_data.append((row['ec_number'], row['sequence']))
    return test_data

###########################################
# Energy Evaluation with PyRosetta
###########################################

class EnergyEvaluator:
    def __init__(self):
        # Initialize PyRosetta (adjust options as needed)
        pyrosetta.init(options="-no_optH false -ignore_unrecognized_res -detect_disulf false")
        self.sfxn = get_score_function(True)

    def evaluate(self, tokens: List[str], pose, pdb_info: str) -> float:
        """
        Evaluate the energy of the last token (fragment) by computing the intra-fragment
        pairwise interaction energy (each unique residue pair within the fragment).
        A lower energy yields a higher reward.
        """
        fragment_lengths = [len(t) for t in tokens]
        fragment_boundaries = []
        start = 1
        for length in fragment_lengths:
            end = start + length - 1
            fragment_boundaries.append((start, end))
            start = end + 1

        if len(fragment_boundaries) < 2:
            return 0.0

        pdb_residues = self._get_residue_chain_map(pdb_info)
        total_fa_atr, total_fa_rep, total_fa_sol = 0.0, 0.0, 0.0

        frag1_start, frag1_end = fragment_boundaries[-2]
        frag2_start, frag2_end = fragment_boundaries[-1]

        frag1_residues = self._get_pose_residues_for_fragment(pdb_residues, frag1_start, frag1_end, pdb_info)
        frag2_residues = self._get_pose_residues_for_fragment(pdb_residues, frag2_start, frag2_end, pdb_info)

        for res1 in frag1_residues:
            for res2 in frag2_residues:
                emap = EMapVector()
                self.sfxn.eval_ci_2b(pose.residue(res1), pose.residue(res2), pose, emap)
                total_fa_atr += emap[fa_atr]
                total_fa_rep += emap[fa_rep]
                total_fa_sol += emap[fa_sol]

        total_energy = total_fa_atr + total_fa_rep + total_fa_sol
        reward = -total_energy
        reward = (reward - (-100)) / (0 - (-100))
        return reward

    def evaluate_test_mode(self, tokens: List[str], pdb_path: str) -> float:
        """
        During testing (when pdb_path is available), compute and print the intra-fragment energy for each token.
        """
        if pdb_path is None:
            print("No pdb_path provided for test energy evaluation.")
            return 0.0

        pose = pose_from_pdb(pdb_path)
        pdb_info = pose.pdb_info()

        fragment_lengths = [len(t) for t in tokens]
        fragment_boundaries = []
        start = 1
        for length in fragment_lengths:
            end = start + length - 1
            fragment_boundaries.append((start, end))
            start = end + 1

        if len(fragment_boundaries) < 2:
            return 0.0

        pdb_residues = self._get_residue_chain_map(pdb_info)
        total_fa_atr, total_fa_rep, total_fa_sol = 0.0, 0.0, 0.0

        for i in range(len(fragment_boundaries) - 1):
            frag1_start, frag1_end = fragment_boundaries[i]
            frag2_start, frag2_end = fragment_boundaries[i + 1]

            frag1_residues = self._get_pose_residues_for_fragment(pdb_residues, frag1_start, frag1_end, pdb_info)
            frag2_residues = self._get_pose_residues_for_fragment(pdb_residues, frag2_start, frag2_end, pdb_info)

            for res1 in frag1_residues:
                for res2 in frag2_residues:
                    emap = EMapVector()
                    self.sfxn.eval_ci_2b(pose.residue(res1), pose.residue(res2), pose, emap)
                    total_fa_atr += emap[fa_atr]
                    total_fa_rep += emap[fa_rep]
                    total_fa_sol += emap[fa_sol]

            total_energy = total_fa_atr + total_fa_rep + total_fa_sol
        reward = -total_energy
        reward = (reward - (-100)) / (0 - (-100))
        return reward

    def _get_residue_chain_map(self, pdb_info):
        pdb_residues = []
        for i in range(1, pdb_info.nres() + 1):
            chain = pdb_info.chain(i)
            residue_number = pdb_info.number(i)
            pdb_residues.append((chain, residue_number))
        return pdb_residues

    def _get_pose_residues_for_fragment(self, pdb_residues, frag_start, frag_end, pdb_info):
        pose_residues = []
        for i, (chain, residue_number) in enumerate(pdb_residues):
            seq_idx = i + 1
            if frag_start <= seq_idx <= frag_end:
                pose_index = pdb_info.pdb2pose(chain, residue_number)
                if pose_index != 0:
                    pose_residues.append(pose_index)
        return pose_residues

###########################################
# Protein Environment for RL
###########################################

class ProteinEnvironment:
    def __init__(self, config: Config, energy_evaluator: EnergyEvaluator):
        self.config = config
        self.energy_evaluator = energy_evaluator
        self.sequence = ""
        self.current_index = 0
        self.done = False
        self.tokens = []
        self.pdb_info = None
        self.pose = None

    def reset(self, sequence: str, pose, pdb_info: str = None):
        """
        For training, a pdb_path is provided.
        For test data, pdb_path is None.
        """
        self.sequence = sequence
        self.current_index = 0
        self.done = False
        self.tokens = []
        self.pdb_info = pdb_info
        self.pose = pose
        return self._get_state()

    def _get_state(self):
        # The current index in the sequence is the state.
        return self.current_index

    def step(self, action: int):
        token_length = action
        next_index = self.current_index + token_length

        if next_index >= len(self.sequence):
            token = self.sequence[self.current_index:]
            self.tokens.append(token)
            self.current_index = len(self.sequence)
            self.done = True
        else:
            token = self.sequence[self.current_index:next_index]
            self.tokens.append(token)
            self.current_index = next_index

        next_state = self._get_state()

        # For test data (when pdb_path is None) set reward to 0.
        if self.pdb_info is None:
            reward = 0
        else:
            reward = self.energy_evaluator.evaluate(self.tokens, self.pose, self.pdb_info) * self.config.reward_scale

        return next_state, reward, self.done

###########################################
# Q-Learning Agent with Normalized Q-Table
###########################################

class QLearningAgent:
    def __init__(self, config: Config):
        self.config = config
        self.q_table = OrderedDict()  # Using an OrderedDict for potential size control.
        self.action_space = self.config.token_sizes

    def _get_q_key(self, state: int, action: int):
        return (state, action)

    def get_best_action(self, state: int):
        q_values = [self.q_table.get(self._get_q_key(state, a), 0.0) for a in self.action_space]
        max_q = max(q_values)
        best_actions = [a for a, qv in zip(self.action_space, q_values) if qv == max_q]
        return random.choice(best_actions)

    def choose_action(self, state: int):
        self.config.epsilon = max(0.1, self.config.epsilon * 0.99)  # Prevent premature convergence
        if random.random() < self.config.epsilon:
            return random.choice(self.action_space)
        else:
            return self.get_best_action(state)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        key = self._get_q_key(state, action)
        old_q = self.q_table.get(key, 0.0)
        if not done:
            next_q_values = [self.q_table.get(self._get_q_key(next_state, a), 0.0) for a in self.action_space]
            max_next_q = max(next_q_values)
            self.config.gamma = max(0.8, self.config.gamma * 0.99)  # Encourage short-term learning
            target = reward + self.config.gamma * max_next_q
        else:
            target = reward
        
        self.config.alpha = max(0.1, self.config.alpha * 0.99)  # Encourage short-term learning
        new_q = old_q + self.config.alpha * (target - old_q)
        self.q_table[key] = new_q

        if len(self.q_table) > self.config.max_q_table_size:
            self._prune_q_table()
    
    def _prune_q_table(self):
        """Prune the Q-table by removing entries with the lowest Q-values."""
        rewards_sum = {key: value for key, value in self.q_table.items()}
        sorted_keys = sorted(rewards_sum, key=lambda k: rewards_sum[k])
        while len(self.q_table) > self.config.max_q_table_size:
            key_to_remove = sorted_keys.pop(0)
            self.q_table.pop(key_to_remove)

    def save(self, filename: str):
        with open(filename, 'w') as f:
            json.dump({str(k): v for k, v in self.q_table.items()}, f)

###########################################
# Training
###########################################

def train(config: Config, sequences_and_pdbs_train: List[Tuple[str, str, str]],
          agent: QLearningAgent, evaluator: EnergyEvaluator):
    """
    Trains the agent on all available PDB data.
    """
    env = ProteinEnvironment(config, evaluator)
    num_epochs = config.num_episodes

    episode_rewards_log = []
    td_errors_log = []
    
    results_dir = "results/tokenization"
    os.makedirs(results_dir, exist_ok=True)

    # Global vocabulary: accumulates tokens from all batches.
    global_vocab = set()

    def plot_metrics(sequence_count, epoch_index):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards_log, label="Episode Reward")
        plt.title(f"Episode Rewards Over Time\nSequences Processed: {sequence_count}")
        plt.xlabel("Sequence Count")
        plt.ylabel("Reward")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(td_errors_log, color="orange", label="TD Error (MSE)")
        plt.title(f"TD Error Over Time\nSequences Processed: {sequence_count}")
        plt.xlabel("Sequence Count")
        plt.ylabel("Mean Squared TD Error")
        plt.legend()

        plt.tight_layout()
        plot_filename = os.path.join(results_dir, f"qlearning_training_plots_epoch_{epoch_index}_seq_{sequence_count}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved training plot after {sequence_count} sequences to {plot_filename}")

    # Main training loop
    for epoch in range(num_epochs):
        sequence_count = 0
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        for _, seq, pdb_path in sequences_and_pdbs_train:
            # if length of sequence is less than 5 or greater than 100, skip
            if len(seq) < 5:
                continue
            if len(seq) > 1024:
                seq = seq[:1024]
            else:
                sequence_count += 1
                pose = pose_from_pdb(pdb_path)
                pdb_info = pose.pdb_info()

                state = env.reset(seq, pose, pdb_info)
                done = False
                seq_reward = 0.0
                td_error_sum = 0.0
                steps_count = 0

                while not done:
                    action = agent.choose_action(state)
                    next_state, reward, done = env.step(action)
                    old_q = agent.q_table.get(agent._get_q_key(state, action), 0.0)
                    agent.update(state, action, reward, next_state, done)
                    new_q = agent.q_table.get(agent._get_q_key(state, action), 0.0)
                    td_error = new_q - old_q
                    seq_reward += reward
                    td_error_sum += (td_error ** 2)
                    steps_count += 1
                    state = next_state

                # Update global vocabulary with tokens from this sequence.
                global_vocab.update(env.tokens)

                episode_rewards_log.append(seq_reward)
                td_errors_log.append(td_error_sum)

                if sequence_count % 10 == 0:
                    plot_metrics(sequence_count, epoch_index=epoch + 1)
            
                    model_filename = os.path.join(results_dir, f"agent_epoch_{epoch + 1}_seq_{sequence_count}.json")
                    agent.save(model_filename)
                    vocab_filename = os.path.join(results_dir, f"global_vocab_epoch_{epoch + 1}_seq_{sequence_count}.json")
                    with open(vocab_filename, 'w') as f:
                        json.dump(list(global_vocab), f)

    print("Training completed.")
    return agent


###########################################
# Applying the Trained Agent to New Sequences
###########################################

def apply_to_new_sequence(config: Config, agent: QLearningAgent, evaluator: EnergyEvaluator,
                          sequences: List[str], pdb_paths: List[str]) -> List[List[str]]:
    """
    For each sequence, the agent tokenizes the sequence.
    If pdb_path is None (as for test data), energy evaluation is skipped.
    """
    env = ProteinEnvironment(config, evaluator)
    all_tokens = []
    for seq, pdb_path in zip(sequences, pdb_paths):
        state = env.reset(seq, pdb_path)
        done = False
        while not done:
            action = agent.get_best_action(state)
            next_state, reward, done = env.step(action)
            state = next_state

        if pdb_path is not None:
            print("\nEvaluating Tokens in Test Mode:")
            evaluator.evaluate_test_mode(env.tokens, pdb_path)
        all_tokens.append(env.tokens)
    return all_tokens

###########################################
# Main Execution
###########################################

if __name__ == "__main__":
    # Step 1: Initialize configuration, evaluator, and agent.
    cfg = Config()
    evaluator = EnergyEvaluator()
    agent = QLearningAgent(cfg)

    # Step 2: create data directory: I have 5 directories and I want to randomely read 200 files from each directory for training and save them to sampled_pbd folder
    pdb_directory = "data/sampled_pdb"
    if not os.path.exists(pdb_directory):
        directories = ["data/metja", "data/human", "data/mouse", "data/ecoli", "data/yeast", "data/danre"] # data are downloaded from alpha fold database
        os.makedirs(pdb_directory, exist_ok=True)
        for directory in directories:
            pdb_files = [f for f in os.listdir(directory) if f.endswith(".pdb")]
            sampled_files = random.sample(pdb_files, 200)
            for pdb_file in sampled_files:
                src = os.path.join(directory, pdb_file)
                dst = os.path.join(pdb_directory, pdb_file)
                os.system(f"cp {src} {dst}")

    sequences_and_pdbs_train = read_pdb_files_from_directory(pdb_directory)
    print(f"Loaded {len(sequences_and_pdbs_train)} PDB files for training.")

    # Step 3: Check if a trained Q-table exists.
    results_dir = "results/tokenization"
    os.makedirs(results_dir, exist_ok=True)
    q_table_path = os.path.join(results_dir, "agent_epoch_5.json")
    if os.path.exists(q_table_path):
        print("Loading trained Q-table...")
        with open(q_table_path, 'r') as f:
            loaded_q = json.load(f)
            agent.q_table = {eval(k): v for k, v in loaded_q.items()}
        print("Q-Table successfully loaded.")
    else:
        print("No trained Q-table found. Training a new agent...")
        agent = train(cfg, sequences_and_pdbs_train, agent, evaluator)