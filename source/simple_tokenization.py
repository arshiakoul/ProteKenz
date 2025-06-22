import os
import random
import json
from collections import OrderedDict
from typing import List, Tuple
import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.core.scoring import fa_atr, fa_rep, fa_sol, EMapVector
import matplotlib.pyplot as plt
import numpy as np

###########################################
# Configuration and Initialization
###########################################

class Config:
    def __init__(self,
                 max_token_size: int = 5,  # Tokens can be between 1-10 AA
                 min_token_size: int = 1,  # Minimum token size
                 num_episodes: int = 10,
                 max_q_table_size: int = 3200,
                 initial_exploration_factor: float = 0.5):  # Initial value for new tokens
        self.max_token_size = max_token_size
        self.min_token_size = min_token_size
        self.num_episodes = num_episodes
        self.max_q_table_size = max_q_table_size
        self.initial_exploration_factor = initial_exploration_factor

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

###########################################
# Energy Evaluation with PyRosetta
###########################################
class EnergyEvaluator:
    def __init__(self):
        # Initialize PyRosetta (adjust options as needed)
        pyrosetta.init(options="-no_optH false -ignore_unrecognized_res -detect_disulf false")
        self.sfxn = get_score_function(True)

    def evaluate(self, tokens: list, pose, pdb_info: str) -> float:
        """
        Evaluate the interaction energy between all token pairs in the sequence.
        """
        # **Step 1: Compute fragment boundaries**
        fragment_lengths = [len(t) for t in tokens]
        fragment_boundaries = []
        start = 1
        for length in fragment_lengths:
            end = start + length - 1
            fragment_boundaries.append((start, end))
            start = end + 1

        # assert length fragment_boundaries == 2
        assert len(fragment_boundaries) == 2

        # **Step 2: Get residue-chain mappings**
        pdb_residues = self._get_residue_chain_map(pdb_info)

        total_fa_atr, total_fa_rep, total_fa_sol = 0.0, 0.0, 0.0

        # **Step 3: Compute energy between two fragments**
        frag1_start, frag1_end = fragment_boundaries[0]
        frag1_residues = self._get_pose_residues_for_fragment(pdb_residues, frag1_start, frag1_end, pdb_info)

        frag2_start, frag2_end = fragment_boundaries[1]
        frag2_residues = self._get_pose_residues_for_fragment(pdb_residues, frag2_start, frag2_end, pdb_info)

        # **Compute pairwise energy for this fragment pair**
        pair_fa_atr, pair_fa_rep, pair_fa_sol = 0.0, 0.0, 0.0
        for res1 in frag1_residues:
            for res2 in frag2_residues:
                emap = EMapVector()
                self.sfxn.eval_ci_2b(pose.residue(res1), pose.residue(res2), pose, emap)
                pair_fa_atr += emap[fa_atr]
                pair_fa_rep += emap[fa_rep]
                pair_fa_sol += emap[fa_sol]

                # Add to total energy
                total_fa_atr += pair_fa_atr
                total_fa_rep += pair_fa_rep
                total_fa_sol += pair_fa_sol

        total_energy = total_fa_atr + total_fa_rep + total_fa_sol
        reward = -total_energy  # Lower energy means higher reward
        reward = (reward - (-100)) / (0 - (-100))
        return reward

    def evaluate_sequence(self, sequence: str) -> float:
        """
        Evaluate the energy of a generated sequence by creating a pose and computing its energy.
        """
        # Create a pose from the sequence
        pose = pose_from_sequence(sequence)
        
        # Score the pose
        score = self.sfxn(pose)
        reward = -score
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
# Q-Learning Agent with Normalized Q-Table
###########################################

class QLearningAgent:
    def __init__(self, config: Config):
        self.config = config
        self.q_table = OrderedDict()  # Square Q-table (tokens as rows & columns)

    def get_token_probabilities(self):
        """
        Convert Q-table values into probabilities for token selection.
        Higher Q-values will result in higher probabilities.
        """
        # Calculate total Q-value for each token using the same approach as sorting
        token_sums = {token: sum(self.q_table[token].values()) for token in self.q_table}
        
        # Normalize values to prevent overflow
        values = np.array(list(token_sums.values()))
        values = values - np.max(values)  # Subtract max for numerical stability
        
        # Convert to probabilities using softmax with numerical stability
        exp_values = np.exp(values)
        probabilities = exp_values / np.sum(exp_values)
        
        # Ensure no NaN values
        probabilities = np.nan_to_num(probabilities, nan=1.0/len(values))
        
        return dict(zip(token_sums.keys(), probabilities))

    def generate_sequence(self, target_length: int) -> str:
        """
        Generate a sequence by sampling tokens from the Q-table.
        """
        sequence = ""
        while len(sequence) < target_length:
            # Select a random token from the Q-table
            token = random.choice(list(self.q_table.keys()))
            sequence += token
            
            # If we've exceeded the target length, trim it
            if len(sequence) > target_length:
                sequence = sequence[:target_length]
                break
                
        return sequence

    def update(self, token1: str, token2: str, reward: float):
        """
        Update the Q-table based on energy-based reconstruction reward.
        """
        if token1 not in self.q_table:
            self.q_table[token1] = {}
        if token2 not in self.q_table[token1]:
            self.q_table[token1][token2] = self.config.initial_exploration_factor

        if token2 not in self.q_table:
            self.q_table[token2] = {}
        if token1 not in self.q_table[token2]:
            self.q_table[token2][token1] = self.config.initial_exploration_factor

        old_q = self.q_table[token1].get(token2)
        new_q = old_q + reward
        self.q_table[token1][token2] = new_q

    def save(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.q_table, f, indent=4)

###########################################
# Training Function
###########################################

def train(config: Config, sequences_and_pdbs: List[Tuple[str, str, str]], agent: QLearningAgent, evaluator: EnergyEvaluator, results_dir: str):
    """Train the agent with dynamic tokenization and Q-table updates."""
    reward_history = []
    
    for epoch in range(config.num_episodes):
        print(f"Starting epoch {epoch + 1}")
        sequence_count = 0
        
        # Process real sequences
        for _, seq, pdb_path in sequences_and_pdbs:
            if len(seq) < 5:
                continue
            if len(seq) > 1024:
                seq = seq[:1024]
    
            pose = pose_from_pdb(pdb_path)
            pdb_info = pose.pdb_info()

            # **Step 1: Generate Random Cuts with Controlled Token Sizes**
            sequence_count += 1
            tokens = []
            current_pos = 0
            while current_pos < len(seq):
                token_length = random.randint(config.min_token_size, config.max_token_size)
                if current_pos + token_length >= len(seq):
                    tokens.append(seq[current_pos:])
                    break
                tokens.append(seq[current_pos:current_pos + token_length])
                current_pos += token_length

            # **Step 3: Compute Energy between New Token and Existing Tokens**
            for i in range(len(tokens)-1):
                for j in range(i+1, len(tokens)):
                    token1, token2 = tokens[i], tokens[j]
                    reward = evaluator.evaluate([token1, token2], pose, pdb_info)
                    agent.update(token1, token2, reward)
                    reward_history.append(reward)  # Track reward
            
            print(f"Processed sequence {sequence_count} with {len(tokens)} tokens")

            # Step 4: Save Model and Metrics every 10 sequences
            if sequence_count % 100 == 0:
                # Generate and evaluate random sequences
                num_generated_sequences = 10  # Adjust this number as needed
                for _ in range(num_generated_sequences):
                    # Generate a random sequence length between 50 and 200
                    target_length = random.randint(50, 200)
                    generated_sequence = agent.generate_sequence(target_length)
                    
                    # Evaluate the generated sequence
                    reward = evaluator.evaluate_sequence(generated_sequence)
                    reward_history.append(reward)
                    
                    # Update Q-table based on the generated sequence's reward
                    tokens = []
                    current_pos = 0
                    while current_pos < len(generated_sequence):
                        token_length = random.randint(config.min_token_size, config.max_token_size)
                        if current_pos + token_length >= len(generated_sequence):
                            tokens.append(generated_sequence[current_pos:])
                            break
                        tokens.append(generated_sequence[current_pos:current_pos + token_length])
                        current_pos += token_length

                    # Update Q-table for token pairs in the generated sequence
                    for i in range(len(tokens)-1):
                        for j in range(i+1, len(tokens)):
                            agent.update(tokens[i], tokens[j], reward)
               
                agent.save(f"{results_dir}/agent_epoch_{epoch + 1}_seq_{sequence_count}_new.json")
                with open(f"{results_dir}/vocabulary_epoch_{epoch + 1}_seq_{sequence_count}_new.txt", 'w') as f:
                    # sort tokens by their sum of Q-values,then save to file
                    token_sums = {token: sum(agent.q_table[token].values()) for token in agent.q_table}
                    sorted_tokens = sorted(token_sums, key=token_sums.get, reverse=True)
                    for token in sorted_tokens:
                        f.write(f"{token}: {token_sums[token]}\n")

                # **Step 5: Plot Reward Trend**
                plt.figure(figsize=(8, 5))
                plt.plot(reward_history, label="Reward Trend")
                plt.xlabel("Updates")
                plt.ylabel("Reward")
                plt.title(f"Epoch {epoch + 1}: Reward Progress")
                plt.legend()
                plt.savefig(f"{results_dir}/reward_epoch_{epoch + 1}.png")
                plt.close()

        print(f"Epoch {epoch + 1} completed.")

    print("Training completed.")
    return agent

###########################################
# Main Execution
###########################################
if __name__ == "__main__":
    cfg = Config()
    evaluator = EnergyEvaluator()
    agent = QLearningAgent(cfg)

    results_dir = "results/simple_tokenization"
    os.makedirs(results_dir, exist_ok=True)

    pdb_directory = "data/sampled_pdb"
    sequences_and_pdbs = read_pdb_files_from_directory(pdb_directory)

    # Load or train Q-table
    q_table_path = os.path.join(results_dir, "agent_latest.json")
    if os.path.exists(q_table_path):
        print("Loading Q-table...")
        with open(q_table_path, 'r') as f:
            agent.q_table = json.load(f)
    else:
        print("Training new agent...")
        agent = train(cfg, sequences_and_pdbs, agent, evaluator, results_dir)