from settings import Settings
from modeling.lda import Corpus, LDAModeler
from modeling.classifier import ClassifierModel, ClassificationDataset
import pandas as pd

class TestLDA:
    def __init__(self):
        self.corpus = Corpus(
                    file_name='test_files/train.csv',
                    language='english',
                    content_column_name=Settings.CONTENT_COL,
                    id_column_name=Settings.ID_COL,
                    phrases_to_join=["anderson cooper", "laura ingraham", "barrack obama"]
                )
        self.lda = LDAModeler(
                    self.corpus,
                    iterations=100,
                    mallet_bin_directory= "c:/mallet/mallet-2.0.8/bin",
                )
    def get_processed_dataset(self):
        return self.corpus.df_docs
    def run_topic_modelling(self):
        a = lda.model_topics()
        return a


class TestClassififer:
    def __init__(self):
        self.labels = ['Politics', '2nd Amendment rights', 'Gun control', 
            'Public opinion', 'Mental health', 'School or public space safety', 
            'Society', 'Race', 'Economic consequences']
        self.num_train_epochs = 3
        self.model_path='distilbert-base-uncased'
        self.train_file='test_files/train_classifier.csv'
        self.dev_file='test_files/dev_classifier.csv'
        self.cache_dir='test_files/cache'
        self.output_dir='test_files/output'
        self.classifier_model = ClassifierModel(
                labels=self.labels,
                num_train_epochs=self.num_train_epochs,
                model_path=self.model_path,
                train_file=self.train_file,
                dev_file=self.dev_file,
                cache_dir=self.cache_dir,
                output_dir=self.output_dir,
            )
    def get_file(self, var):
        if(var=='train'):
            fi = pd.read_csv(self.train_file)
        else:
            fi = pd.read_csv(self.dev_file)
        cd = self.classifier_model.make_dataset(fi, 'Example', 'Category')
        return cd
    def train_and_evaluate(self):
        metrics = self.classifier_model.train_and_evaluate()
        return metrics
    def do_cross_validation(self):
        return self.classifier_model.perform_cv_and_train()

# lda_in = TestLDA()
# print(lda_in.get_processed_dataset())

classifier = TestClassififer()
# print(classifier.train_and_evaluate())
# print(classifier.get_file('dev'))
print(classifier.do_cross_validation())