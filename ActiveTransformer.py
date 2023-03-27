from tqdm import tqdm
import os
from datetime import datetime
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from codecarbon import EmissionsTracker
import torch
import random

class ActiveLearner():
    def __init__(self,
                 model_name:str,
                 data_name:str,
                 emission_path:str,
                 strategy:str,
                 initial_size:float = 0.05,
                 train_size:float=.7, 
                 pool_reduction:float=0.0,
                 batchsize:int=32, 
                 epochs:int=3,
                 seed:int=2023) -> None:
        self.model_name = model_name
        self.data_name = data_name
        self.model_ID, self.data_path = self.select_model_and_data(model_name, data_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ID)
        self.set_training_args(batchsize=batchsize, epochs=epochs)
        self.seed = seed
        self.emission_path = emission_path
        self.train_size = train_size
        self.initial_size = initial_size
        self.pool_reduction = pool_reduction
        self.counter = -1
        self.make_pool_and_test()
        self.num_labels = self.df.label.unique().shape[0]
        self.full_size = len(self.df)
        self.query_strategy = self.choose_query_strategy(strategy)
    
    def make_baseline(self, model_name, data_name):
        now = datetime.now()
        model_ID, data_path = self.select_model_and_data(model_name, data_name)
        df = pd.read_csv(data_path)
        tokenizer = AutoTokenizer.from_pretrained(model_ID)
        df = self.df[['text', 'label']]
        df['label'] = df['label'].astype(int)
        df = df.sample(frac=1, random_state=self.seed + 1)
        train_df = df.groupby('label').sample(frac=self.train_size, random_state=self.seed)
        test_df = df.drop(train_df.index)
        train_df, test_df = train_df.sample(frac=1, random_state=self.seed + 1), test_df.sample(frac=1, random_state=self.seed + 1)
        train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)
        traindata, testdata = self.turn_to_dataset(train_df), self.turn_to_dataset(test_df)
        model = AutoModelForSequenceClassification.from_pretrained(model_ID, num_labels=self.num_labels)
        trainer = Trainer(model=model,args=self.training_args,train_dataset=traindata)
        tracker = EmissionsTracker(project_name=f'Training_{self.data_name}_BASELINE', output_dir=self.emission_path)
        tracker.start() 
        trainer.train()
        tracker.stop()
        emission_df = pd.read_csv(f'{self.emission_path}/emissions.csv')
        train_energy = float(emission_df.loc[emission_df.project_name == f'Training_{data_name}_BASELINE', 'energy_consumed'].iloc[-1])
        train_duration = float(emission_df.loc[emission_df.project_name == f'Training_{data_name}_BASELINE', 'duration'].iloc[-1])
        y_pred = np.argmax(trainer.predict(testdata).predictions, axis=1)
        y_true = np.array(testdata['label'])
        f1 = np.round(f1_score(y_true, y_pred, average='weighted'),2)
        accuracy =np.round(accuracy_score(y_true, y_pred),2)
        recall = np.round(recall_score(y_true, y_pred, average='weighted'),2)
        precision = np.round(precision_score(y_true, y_pred, average='weighted'),2)
        return pd.DataFrame(dict(
            F1 = [f1],
            Accuracy = [accuracy],
            Recall = [recall],
            Precision = [precision],
            Data = [data_name],
            Model = [model_name],
            Trainset_absolute = [len(train_df)],
            Test_absolute = [len(testdata)],
            Train_energy = [train_energy],
            Train_duration = [train_duration],
            Datetime = [now.strftime("%Y-%m-%d %H:%M:%S")],
            ))
    def make_pool_and_test(self):
        self.df = pd.read_csv(self.data_path)
        self.df = self.df[['text', 'label']]
        self.df['label'] = self.df['label'].astype(int)
        self.df = self.df.sample(frac=1, random_state=self.seed + 1)
        self.pool_df = self.df.groupby('label').sample(frac=self.train_size, random_state=self.seed)
        self.test_df = self.df.drop(self.pool_df.index)
        self.pool_df, self.test_df = self.pool_df.sample(frac=1-self.pool_reduction, random_state=self.seed + 1), self.test_df.sample(frac=1, random_state=self.seed + 1)
        self.pool_df, self.test_df = self.pool_df.reset_index(drop=True), self.test_df.reset_index(drop=True)
        self.testdata = self.turn_to_dataset(self.test_df)
        
    def set_training_args(self,batchsize:int=64, epochs:int=3):
        self.training_args = TrainingArguments(output_dir="TransformerActive", 
                            #fp16=True,
                            #optim="adafactor",
                            do_eval=False, 
                            per_device_train_batch_size=batchsize, 
                            per_device_eval_batch_size=batchsize,
                            num_train_epochs=epochs)
        
    def select_model_and_data(self, model_name, data_name):
        data_set_names = {
            'ClaimDetection':'/home/sami/READER_REPO/Experiments/ActiveLearning/data/GermevalFactClaiming.csv', #https://github.com/germeval2021toxic/SharedTask/tree/main/Data%20Sets
            '10k_Newsarticles' : '/home/sami/READER_REPO/Experiments/ActiveLearning/data/10k_newsarticles.csv', #https://github.com/tblock/10kGNAD
            'Claimbuster':'/home/sami/READER_REPO/Experiments/ActiveLearning/data/Claimbuster.csv', #https://zenodo.org/record/3836810#.Y_ZDNrSZOvB
            'imdb' : '/home/sami/READER_REPO/Experiments/ActiveLearning/data/imdb.csv', #https://huggingface.co/datasets/imdb
            'Twitter_Sentiment': '/home/sami/READER_REPO/Experiments/ActiveLearning/data/Twitter_Sentiment.csv', #https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis
            'Liar' : '/home/sami/READER_REPO/Experiments/ActiveLearning/data/Liar.csv', #https://huggingface.co/datasets/liar
            'AG_News' : '/home/sami/READER_REPO/Experiments/ActiveLearning/data/AG_News.csv', #https://huggingface.co/datasets/ag_news
            'Go_Emotions' : '/home/sami/READER_REPO/Experiments/ActiveLearning/data/Go_Emotions.csv', #https://huggingface.co/datasets/go_emotions/viewer/simplified/train
            'MedicalAbstracts' : '/home/sami/READER_REPO/Experiments/ActiveLearning/data/MedicalAbstracts.csv',#https://github.com/sebischair/Medical-Abstracts-TC-Corpus
            'PatientReviews' : '/home/sami/READER_REPO/Experiments/ActiveLearning/data/PatientReviews.csv',#https://www.kaggle.com/datasets/thedevastator/german-2021-patient-reviews-and-ratings-of-docto
            'Yahoo': '/home/sami/READER_REPO/Experiments/ActiveLearning/data/Yahoo.csv', #https://www.kaggle.com/datasets/yacharki/yahoo-answers-10-categories-for-nlp-csv
            'Cola' : '/home/sami/READER_REPO/Experiments/ActiveLearning/data/COLA.csv', #https://nyu-mll.github.io/CoLA/
            }
        transformer_names = {
            'GBert-Base' : 'deepset/gbert-base',
            'GBert-Large' : 'deepset/gbert-large',
            'GElectra-Base' : 'deepset/gelectra-base',
            'GElectra-Large' : 'deepset/gelectra-large',
            'XLMR-Base' : 'xlm-roberta-base',
            'XLMR-Large' : 'xlm-roberta-large',
            'Bert-Base':'bert-base-uncased'
            }
        model_ID = transformer_names[model_name]
        data_path = data_set_names[data_name]
        return model_ID, data_path
            
    def root_sample(self):
        if self.initial_size > 1:
            self.root_idxs = list(self.pool_df.sample(n=self.initial_size, random_state=self.seed).index)
        else:
            self.root_idxs = list(self.pool_df.sample(frac=self.initial_size, random_state=self.seed).index)
        
    def choose_query_strategy(self, strategy:str):
        def uncertainty_sampling(pred_probs, sample_size):
            uncertainty = 1 - softmax(pred_probs, axis=1).max(axis=1)
            idxs = np.argsort(uncertainty)[-sample_size:]
            return list(idxs)

        def classification_margin(pred_probs, sample_size):
            part = np.partition(-pred_probs, 1, axis=1)
            margin = - part[:, 0] + part[:, 1]
            idxs = np.argsort(margin)[:sample_size]
            return list(idxs)

        def classification_entropy(pred_probs, sample_size):
            e = entropy(pred_probs.T)
            idxs = np.argsort(e)[-sample_size:]
            return list(idxs)
        
        def random_baseline(query_df, sample_size):
            random.seed(self.seed)
            seq = list(range(query_df.shape[0]))
            idxs = random.sample(seq, sample_size)
            return list(idxs)
        
        query_dict = {
            'Least_Confidence':uncertainty_sampling,
            'Breaking_Ties':classification_margin,
            'Prediction_Entropy':classification_entropy,
            'Random_Sampling':random_baseline
            }
        self.strategy = strategy
        return query_dict[strategy]
        
    def query(self, sample_size, idxs = None):
        now = datetime.now()
        if idxs == None:
            self.root_sample()
            idxs = self.root_idxs
        train_df =  self.pool_df.iloc[idxs,:]
        query_df = self.pool_df.drop(idxs)
        train_size = len(train_df)
        trainer, train_energy, train_duration = self.train(train_df)
        new_idxs, query_energy, query_duration = self.active_sampling(trainer, query_df, sample_size)
        idxs = idxs + new_idxs
        f1, accuracy, recall, precision, test_energy = self.test(trainer)
        results = dict(
            F1 = f1,
            Accuracy = accuracy,
            Recall = recall,
            Precision = precision,
            Query = self.strategy,
            Data = self.data_name,
            Dataset_size = self.full_size,
            Model = self.model_name,
            Round = self.counter,
            Sample_size = sample_size,
            Num_Labels = self.num_labels,
            Trainset_absolute = train_size,
            Query_absolute = len(query_df),
            Test_absolute = len(self.testdata),
            Percent = round((train_size/len(self.pool_df))*100,2),
            Train_energy = train_energy,
            Query_energy = query_energy,
            Test_energy = test_energy,
            Train_duration = train_duration,
            Query_duration = query_duration,
            Train_distribution = list(train_df.label.value_counts()/len(train_df)),
            Query_distribution = list(self.pool_df.iloc[new_idxs,:].label.value_counts()/len(self.pool_df.iloc[new_idxs,:])),
            Test_distribution = list(self.test_df.label.value_counts()/len(self.test_df)),
            Datetime = now.strftime("%Y-%m-%d %H:%M:%S"),
        )
        print(json.dumps(results, indent=4))
        try:
            with open("/home/sami/READER_REPO/Experiments/ActiveLearning/scripts/ActiveCronProtocol.txt","a") as file:
                    file.write('\n')
                    file.write(results)
        except Exception:
            pass
        return idxs, results
        
    def train(self, train_df):
        self.counter += 1
        traindata = self.turn_to_dataset(train_df)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_ID, num_labels=self.num_labels)
        trainer = Trainer(model=model,args=self.training_args,train_dataset=traindata)
        tracker = EmissionsTracker(project_name=f'Training_{self.counter}_{self.data_name}_{self.strategy}', output_dir=self.emission_path)
        tracker.start() 
        trainer.train()
        tracker.stop()
        df = pd.read_csv(f'{self.emission_path}/emissions.csv')
        train_energy = float(df.loc[df.project_name == f'Training_{self.counter}_{self.data_name}_{self.strategy}', 'energy_consumed'].iloc[-1])
        train_duration = float(df.loc[df.project_name == f'Training_{self.counter}_{self.data_name}_{self.strategy}', 'duration'].iloc[-1])
        return trainer, train_energy, train_duration
    
    def active_sampling(self, trainer, query_df, sample_size):
        querydata = self.turn_to_dataset(query_df)
        tracker = EmissionsTracker(project_name=f'Query_{self.counter}_{self.data_name}_{self.strategy}', output_dir=self.emission_path)
        tracker.start()
        if self.strategy == 'Random_Sampling':
            pred_probs = query_df 
        else:
            pred_probs = trainer.predict(querydata).predictions
        tracker.stop()
        df = pd.read_csv(f'{self.emission_path}/emissions.csv')
        query_energy = float(df.loc[df.project_name == f'Query_{self.counter}_{self.data_name}_{self.strategy}', 'energy_consumed'].iloc[-1])
        query_duration = float(df.loc[df.project_name == f'Query_{self.counter}_{self.data_name}_{self.strategy}', 'duration'].iloc[-1])
        sample_idxs = self.query_strategy(pred_probs, sample_size)
        frame_idxs = [list(query_df.index)[idx] for idx in sample_idxs]
        return frame_idxs, query_energy, query_duration
    
    def test(self, trainer):
        tracker = EmissionsTracker(project_name=f'Test_{self.counter}_{self.data_name}_{self.strategy}', output_dir=self.emission_path)
        tracker.start() 
        y_pred = np.argmax(trainer.predict(self.testdata).predictions, axis=1)
        tracker.stop()
        y_true = np.array(self.testdata['label'])
        df = pd.read_csv(f'{self.emission_path}/emissions.csv')
        test_energy = float(df.loc[df.project_name == f'Test_{self.counter}_{self.data_name}_{self.strategy}', 'energy_consumed'].iloc[-1])
        f1 = np.round(f1_score(y_true, y_pred, average='weighted'),2)
        accuracy =np.round(accuracy_score(y_true, y_pred),2)
        recall = np.round(recall_score(y_true, y_pred, average='weighted'),2)
        precision = np.round(precision_score(y_true, y_pred, average='weighted'),2)
        del trainer
        return f1, accuracy, recall, precision, test_energy
    
    def active_learning(self, sample_size:int, sample_up_to:float):
        pool_size = len(self.pool_df)
        pool_size *= (1-self.initial_size)
        pool_size *= sample_up_to
        sampling_steps = int(pool_size//sample_size) + 1
        idxs = None
        self.result_list = list()
        for sampling_round in range(sampling_steps):
            idxs, results = self.query(idxs=idxs, sample_size= sample_size)
            self.result_list.append(results)
        self.results = pd.DataFrame(self.result_list)
        #self.pool_df.iloc[idxs,:].to_csv(f'/home/sami/READER_REPO/Stats/Data/ActiveLearner/Datasplits/ACTIVE_TRAIN_{self.data_name}_{self.strategy}.csv', index = False)
        return self.results
        
    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], padding="max_length", truncation=True)
    
    def turn_to_dataset(self, df):
        return Dataset.from_pandas(df).map(self.tokenize_function, batched=True)

def assign_size(data_name):
    dct = {'GermevalFactClaiming':.1,
        'Cola':.1,
        'Subjectivity':.1,
        '10k_Newsarticles':.1,
        'Liar':.1,
        'MedicalAbstracts':.1,
        'Go_Emotions':.1,
        'Hatespeech':.1,
        'Claimbuster':.1,
        'imdb':.07,
        'PatientReviews':.07,
        'filmstarts':.07,
        'Yahoo':.05,
        'AG_News':.05,
        'Twitter_Sentiment':.03}
    return dct[data_name]

def assign_bert(data_name):
    dct = {'GermevalFactClaiming': 'deepset/gbert-base',
        'COLA':'bert-base-uncased',
        'Subjectivity':'bert-base-uncased',
        '10k_Newsarticles': 'deepset/gbert-base',
        'Liar':'bert-base-uncased',
        'MedicalAbstracts':'bert-base-uncased',
        'Go_Emotions':'bert-base-uncased',
        'Hatespeech':'bert-base-uncased',
        'Claimbuster':'bert-base-uncased',
        'imdb':'bert-base-uncased',
        'PatientReviews': 'deepset/gbert-base',
        'filmstarts':'bert-base-uncased',
        'Yahoo':'bert-base-uncased',
        'AG_News':'bert-base-uncased',
        'Twitter_Sentiment':'bert-base-uncased'}
    return dct[data_name]
    
if __name__ == '__main__':
    queries = ['Random_Sampling', 'Prediction_Entropy', 'Breaking_Ties']
    datasets = ['Liar', 'MedicalAbstracts', 'Go_Emotions', 'imdb', 'Cola','PatientReviews', 'AG_News', 'Yahoo','Twitter_Sentiment']
    for IDX, dataset in enumerate(datasets):
        for pool_reduction in [.5, .3]:
            for idx, query_strategy in enumerate(queries): 
                print(f'/home/sami/READER_REPO/Stats/Data/ActiveLearner/Pool{int((1-pool_reduction)*100)}')                    
                try:
                    model_name = 'XLMR-Base'
                    strategy = query_strategy
                    data_name = dataset
                    emission_path = 'EMISSION_PATH'
                    pool_reduction = pool_reduction
                    initial_size = 0.01
                    learner = ActiveLearner(
                                    model_name=model_name, 
                                    data_name=data_name,
                                    emission_path=emission_path,
                                    strategy=strategy,
                                    initial_size = initial_size,
                                    pool_reduction = pool_reduction
                                    )
                    sample_up_to = assign_size(data_name)
                    sample_size = 50
                    results = learner.active_learning(sample_size, sample_up_to)
                    results.to_csv(f'/PATH/Pool{int((1-pool_reduction)*100)}/Results_{query_strategy}_{dataset}_{sample_size}.csv', index = False)
                except Exception as e:
                    print(e)
