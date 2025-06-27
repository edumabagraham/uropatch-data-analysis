### The folder named feature_based_data has all the extracted features. Each file is named according to the type of features which have been extracted.

#### three_class_(raw/up)_(1s/2s/5s).csv

raw - the data was not over-sampled before the features were extracted.

up - the data was over-sampled before the features were extracted.

1s - features were extracted with a 1 second window

2s - features were extracted with a 2 second window

5s - features were extracted with a 5 second window

#### three_class_(raw/up)_(1s/2s)_ag.csv
all the above rules apply and the gyroscope features were extracted.


### svm_pretrained_model

This contains the model trained in Google Co-lab.


## Try out 
To run the models with different sets of features, put in the path to the desired feature set located in the feature_based_data folder.