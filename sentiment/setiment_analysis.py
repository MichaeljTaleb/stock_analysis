# Credit: https://www.kdnuggets.com/how-to-fine-tune-bert-sentiment-analysis-hugging-face-transformers & Michael Taleb
# email: michael.j.taleb@vanderbilt.edu
# Description: this program is a component of the APIs/data_analysis file
# it is written in a separate program so that it is easier to debug and write
# it will be pasted into APIs/data_analysis analyse_article function after completion and testing
# the purpose of this program is to take in an article as a string parameter and return whether or not
# it indicates to buy or sell a stock

# Using a virtual environment for dependencies:
# active in terminal by "source venv/bin/activate" & "deactivate"


# import sys
# print(sys.executable)

from datasets import load_dataset
from transformers import BertTokenizer

def tokenize_function(dataset):
    # return tokenizer(examples['text'], padding="max_length", truncation=True)
    # max_length is 512, reduced to save time and space.
    return tokenizer(dataset['text'], padding="max_length", truncation=True, max_length=128)


def print_dataset(tokenized_datasets):
    tokenized_split = tokenized_datasets['train']
    # Validating dataset
    print(len(tokenized_split))
    for i in range(len(tokenized_split)):
        print(tokenized_split[i])
        print(i)
        print()




dataset = load_dataset("csv",
                       data_files='/Users/michaeltaleb/Documents/archive/all-data.csv',
                       encoding='ISO-8859-1',
                       column_names=['label', 'text'])
dataset_split = dataset['train']
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_datasets = dataset.map(tokenize_function, batched=True)



# train_testvalid = tokenized_datasets['train'].train_test_split(test_size=0.2)
# train_dataset = train_testvalid['train']
# valid_dataset = train_testvalid['test']
#
# print(train_dataset)
# print(valid_dataset)
#
# from torch.utils.data import DataLoader
#
# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
# valid_dataloader = DataLoader(valid_dataset, batch_size=8)
#
#
# # Fine-tuning and farther training BERT
# from transformers import BertForSequenceClassification, AdamW
#
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
#
# from transformers import Trainer, TrainingArguments
#
# training_args = TrainingArguments(
#     output_dir='./results',
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=valid_dataset,
# )
#
# trainer.train()
#
# # Evaluating Metrics
# metrics = trainer.evaluate()
# print(metrics)
#
# # Predicting
# predictions = trainer.predict(valid_dataset)
# print(predictions)
