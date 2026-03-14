# blur_obfuscation_project
GENERATIVE OPTICAL OBFUSCATION THROUGH NATURAL EXPOSURE RECONSTRUCTION


Formatting Instructions:
Save all model weights, but not on github. Github will follow this structure:
 Epic {epic_number}
EX_{id_from_sheet}
Raw/
Any datasets generated, if any
Model/
Checkpoints here (don’t push to git)
Results/
Preds.csv (if applicable)
Graphs
Any graphs here
Logs/
README.md
EX_{experiment_id_from_sheet}_main.ipynb
Any other files/folders desired


Use train_model function from utils.py (in the main directory) whenever possible. If not make sure you log everything it does with w&b too, and have the same metrics

As much as possible, put all vary-able parameters in the experiment’s specific config.yaml in the /configs file

As much as possible, use the evaluate function from configs, and the dataset. its built to work for as much as it can and help with standardization.
