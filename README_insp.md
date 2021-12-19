# InSearchOfHiggsBoson

## Team Members

Diogo Soares : diogo.sousasoares@epfl.ch

Saad El Moutaouakil : mohamed.elmoutaouakil@epfl.ch

Gonçalo Gomes : goncalo.cavacogomes@epfl.ch

## Purpose
This project tries to emulate the work done by the CERN team responsible for finding the boson of Higgs.

## How to Run

Clone the project in your local machine, and uncompress all the datasets contained in /data repository.

Run the file "run.py" using a python interpreter

## Project Sctruture

In file *run.py* one can find the normal project execution, which predicts the presence or not of the boson of Higgs.

In file *implementations.py* one can find the machine learning models used in this project. 

In file *auxiliary_to_implementations.py* one can find the helper functions used to build the models defined above.

In file *proj1_helpers.py* one can find helper functions for the normal execution of the project.

## Run Structure

###### Loading data

Load function from CERN dataset

###### Split data in 3 groups

Split data in 3 groups according to the number of jets, this division is motivated by the different physical phenomena.

###### Clean data

Remove nan values and replace by the mean or max.
Remove features with variance 0.

###### Process data

Transform data using standard procedures, such as centering


Add polynomial expansion to be able to model functions with higher polynomial degree.

###### Train Ridge Regression to models

Perform a standard ridge regression to every model.

###### Test Models in training data

Output some indicative values of the results of the models in labeled data (not used for testing).

###### Generate Predictions

Generate predictions for data with unknown labels.

###### Create Submission file 

Create a submission file for grading in AIcrowd.




