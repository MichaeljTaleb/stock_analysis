# Credit: https://www.kdnuggets.com/how-to-fine-tune-bert-sentiment-analysis-hugging-face-transformers & Michael Taleb
# email: michael.j.taleb@vanderbilt.edu
# Description: this program is a component of the APIs/data_analysis file
# it is written in a separate program so that it is easier to debug and write
# it will be pasted into APIs/data_analysis analyse_article funciton after completion and testing
# the purpose of this program is to take in an article as a string parameter and return whether or not
# it indicates to buy or sell a stock

# Using a virtual environment for dependencies:
# active in terminal by "source venv/bin/activate" & "deactivate"




import sys
print(sys.executable)
#
# from datasets import load_dataset
#
# dataset = load_dataset("imdb")
# print(dataset)
#
# from transformers import BertTokenizer
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
#
# def tokenize_function(examples):
#     return tokenizer(examples['text'], padding="max_length", truncation=True)
#
#
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
#
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
