ProteKenz

ProteKenz is a protein tokenization and sequence representation project developed for research in machine learning–driven drug discovery. It focuses on experimenting with various tokenization methods—including fixed k-mers, subword encodings, and boundary-aware slicing—on protein sequences extracted from PDB files.

Project Purpose

This project evaluates how different tokenization strategies impact sequence representation quality, especially in low-data settings. It supports tasks like protein classification, structure prediction, and embedding learning with shallow neural encoders.

Folder Overview

data/sampled_pdb – Input directory for .pdb files

source/simple_tokenization.py – Main tokenization script

results/ – Output tokens and logs

notebooks/ – Jupyter notebooks for analysis and visualization

Requirements

Python 3.8

PyRosetta

SentencePiece (for BPE/Unigram)

NumPy, pandas, matplotlib

Usage

Place PDB files in the data/sampled_pdb folder. Then run the script from the project root. Tokenized outputs will be saved under results/.

Author

Arshia Koul
Research Assistant – University of Colorado Denver
arshiakoul@gmail.com
