﻿# Finding meaning in autogenerated text
 ## Team Members
 
 Gonçalo Gomes : goncalo.cavacogomes@epfl.ch
 
 Erik Börve :
 
 Marcus HenriksbØ:
 
 ## Purpose
 This project tries to focus on spoken language understanding aiming to derive the meaning of users queries, to help conversational agent to respond appropriately to the user, and to evaluate the performance and the impact of using automatically generated transcripts using audio speech recognitian engine on the intent classification task.
 
 ## How to Run
 Clone the project in your local machine, and uncompress all the datasets contained in /snips repository
 
 Run each .ipynb file using jupyter notebook interpreter
 
 ## Project Structure
 
 In file *Wav2vec.ipynb* one can find the Audio speech recognition engine used to produce the automatically generated transcripts.
 In file *Handling_Data.ipynb* one can find the dataset manipulation using pandas dataframes to produce datasets properly fitted to use on the natural language processing technics and on the classification tasks
 In file *Bert.ipynb* one can find the normal classification task execution using Bert's natural language processing techniques to generate the feature space
 In file *TfIdf.ipynb* one can find the normal classification task execution using TfIdf's natural language processing techniques to generate the feature space
 In file *Word2Vec.ipynb* one can find the normal classification task execution using Word2Vec's natural language processing techniques to generate the feature space
 In file *_helpers.py* one can find the helper functions for the normal execution of the project.
 
 ## Run Structure
 
 ### For Speach Recognition (Wav2vec.ipynb)
 
 #### Loading data
 Load .wav files
 
 #### Loading models
 Load facebook's pre-trained speech recognition models
 
 #### Audio files properties check
 Check the sampling rate of audio files
 
 #### Audio speech recognition engine
 Transform the audio into string
 
 #### Auto-correction
  Do an individually word autocorrection for each word present in ASR transcripts for the most similar word present in the Groundtruth word vocabulary.
 #### Save dataframe
  Save ASR data into a .csv file
  
 ### For Data manipulation (Handling_Data.ipynb)
  
 #### Loading data
  Load data set
 #### Label data
  Label each transcript to the corresponding intent index 
    
 ### Normal Execution
  This structure is repeated on every classification task files
 
 #### Loading Data
  Load datasets properly fitted to use in the natural language processing methods and classification tasks
 
 #### Feature Space generation
  Produce the feature space using the proper Natural Language Processing Technique
 
 #### Split into test/train 
  Split data into train and test samples to use in the classification tasks
 
 #### Generate Predictions
  Generate predictions for each classifier
  
 #### Model Evaluation
  Evaluate each classifier model using the proper metrics
 
 #### Try your self
  Final function in the file focusing to try catch the user intent in new made up sentences with the NLP technique used in the file and a given classifier.
  
 
 
 
 
 
