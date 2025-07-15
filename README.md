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

| Number | Feature set filename  | Description |
|--------| ------------- | ------------- |
|1 | two_class_raw_1s_no.csv  | Void and non-void classes. Features are extracted with a 1 second sliding window with no overlap  |
|1b | two_class_raw_1s_yo_0.5.csv  | Void and non-void classes. Features are extracted with a 1 second sliding window with 0.5 overlap  |
|1c |two_class_raw_1s_yo_0.8.csv|Void and non-void classes. Features are extracted with a 1 second sliding window with 0.8 overlap |
|2 | two_class_raw_2s_no.csv | Void and non-void classes. Features are extracted with a 2 second sliding window with no overlap |
|2b | two_class_raw_2s_yo_0.5.csv | Void and non-void classes. Features are extracted with a 2 second sliding window with 0.5 overlap |
|2c | two_class_raw_2s_yo_0.8.csv | Void and non-void classes. Features are extracted with a 2 second sliding window with 0.8 overlap |
|3 | two_class_raw_3s_no.csv | Void and non-void classes. Features are extracted with a 3 second sliding window with no overlap |
|3b | two_class_raw_3s_yo_0.5.csv | Void and non-void classes. Features are extracted with a 3 second sliding window with 0.5 overlap |
|3c | two_class_raw_3s_yo_0.8.csv | Void and non-void classes. Features are extracted with a 3 second sliding window with 0.8 overlap |
|4 | two_class_raw_4s_no.csv | Void and non-void classes. Features are extracted with a 3 second sliding window with no overlap |
|4b | two_class_raw_4s_yo_0.5.csv | Void and non-void classes. Features are extracted with a 3 second sliding window with 0.5 overlap |
|4c | two_class_raw_4s_yo_0.8.csv | Void and non-void classes. Features are extracted with a 3 second sliding window with 0.8 overlap |
|5 | two_class_raw_5s_no.csv | Void and non-void classes. Features are extracted with a 3 second sliding window with no overlap |
|5b | two_class_raw_5s_yo_0.5.csv | Void and non-void classes. Features are extracted with a 3 second sliding window with 0.5 overlap |
|5c | two_class_raw_5s_yo_0.8.csv | Void and non-void classes. Features are extracted with a 3 second sliding window with 0.8 overlap |