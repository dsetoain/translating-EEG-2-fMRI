# Thesis-work
All the coding behind the thesis project (matlab script included)

The encoder_decoder file holds the translation model, it is run through the training.py file which handles loading the input data, running the model, training the model on the data, and printing the values for MSE and Pearson used in the paper. Also runs the plotting but this part is commented out in this version. 

The Permutation.py file runs the permutation-based importance test used in the study, the subject and session to do can be changed in the parameters.

The matlab_script should be run in matlab... and it handles converting the EEG data extracted from the NATVIEW project into the feature matrix (both versions can be seen in the EEG file with the fMRI files but it is so big I could only send the first three examples)

The outputs folder has the resulting true and predicted signals made from the model in csv and npy form.

The models folder has the intermediate models made so the training does not have to be repeated for the permutation based importance test. 

The EEG_data folder has the first three subjects data including the EEG file used, the fMRI file used (the 100 parcellation one) and the matrix for the EEG input (i think the structure from this one drastically changed when i put it here so idk if it will work well... let me know and I can send you the actual feature matrix)
