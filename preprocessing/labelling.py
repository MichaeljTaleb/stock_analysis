# Author: Michael Taleb
# email: michael.j.taleb@vanderbilt.edu
# Description: The sentiment program requires the labels to be in -1, 0, 1 rather than positive, negative and zero

from datasets import load_dataset
import pandas as pd


# This prints a dataset that is already split into train and test
# If you want to print a dataset before the split that it must be split here in the function
def print_dataset(tokenized_datasets):
    tokenized_train = tokenized_datasets['train']
    print(len(tokenized_train))
    for i in range(20):
        print(tokenized_train[i])
        print(i)
        print()


# Tokenizing dataset

dataset = load_dataset("csv",
                       data_files='/Users/michaeltaleb/Documents/archive/tokenized_labelled_data.csv',
                       encoding='ISO-8859-1',
                       column_names=['label', 'text'])



# Switching labels from neg, pos, and neut to int values
labels = { "negative" : -1, "neutral" : 0, "positive" : 1}

def convert_labels(row):
    row['label'] = labels[row['label']]
    return row

dataset = dataset.map(convert_labels, batched=False)
# Convert to pandas DataFrame and save to CSV
df = pd.DataFrame(dataset['train'])
df.to_csv('/Users/michaeltaleb/Documents/archive/converted_labels.csv', index=False)

print("done")


dataset_labelled = load_dataset("csv",
                       data_files='/Users/michaeltaleb/Documents/archive/converted_labels.csv',
                       encoding='ISO-8859-1',
                       column_names=['label', 'text'])

print_dataset(dataset_labelled)





