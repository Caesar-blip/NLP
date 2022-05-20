Read for code submission group project 2
Nils Breeman, Sebastiaan Bye, Julius Wantenaar

code_part_a.ipynb contains answers to question 1, 2, 3 and 4.
BERT was trained locally on a GPU so torch.cuda.is_available() should return true to succesfully run the code


code_part_b.ipynb contains answers to question 5, 6 and 7.
BERT is loaded in using joblib.load, for this to work BERT should be trained using the code in part a