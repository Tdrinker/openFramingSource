"""Classifier related backend functionality."""
import tempfile
import typing as T

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import typing_extensions as TT
from sklearn.metrics import classification_report  # type: ignore
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataset import Dataset
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EvalPrediction
from transformers import InputFeatures  # type: ignore
from transformers import Trainer
from transformers import TrainingArguments  # type: ignore
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_utils import PredictionOutput
from flask import current_app as app
from flask_app.modeling.lda import CSV_EXTENSIONS
from flask_app.settings import Settings

# from modeling.lda import CSV_EXTENSIONS
# from settings import Settings
import os
# Named as such to distinguish from db.ClassifierMetrics
ClassifierMetricsJson = TT.TypedDict(
    "ClassifierMetricsJson",
    {
        "accuracy": float,
        "macro_f1_score": float,
        "macro_recall": float,
        "macro_precision": float,
    },
)


class ClassificationDataset(Dataset):  # type: ignore
    """Inherits from Torch dataset. Loads and holds tokenized data for a BERT model."""

    def __init__(
        self,
        labels: T.List[str],
        tokenizer: PreTrainedTokenizer,
        label_map: T.Dict[str, int],
        dset_filename: str,
        content_column: str,
        label_column: T.Optional[str],
    ):
        """.

        labels: list of valid labels (can be strings/ints)
        tokenizer: AutoTokenizer object that can tokenize input text
        label_map: maps labels to ints for machine-readability
        dset_filename: name of the filename (full filepath) of the dataset being loaded
        content_column: column name of the content to be read
        label_column: column name where the labels can be found
        """
        suffix = dset_filename.split(".")[-1]  # type: ignore

        if suffix in CSV_EXTENSIONS:
            doc_reader = lambda b: pd.read_csv(b, dtype=object)
        else:
            raise ValueError(
                f"The file {dset_filename} doesn't have a recognized extension."
            )

        self.labels = labels
        self.label_map = label_map
        self.tokenizer = tokenizer
        df = doc_reader(dset_filename)  # type: ignore
        self.len_dset = len(df)

        self.content_series = df[
            content_column
        ]  # For later, if we need to output predictions
        self.encoded_content = self.tokenizer.batch_encode_plus(
            df[content_column], max_length=None, pad_to_max_length=True,
        )
        if label_column is not None:
            self.encoded_labels: T.Optional[T.List[int]] = [
                self.label_map[label] for label in df[label_column]
            ]
        else:
            self.encoded_labels = None
        self.features = []
        for i in range(len(self.encoded_content["input_ids"])):
            inputs = {
                k: self.encoded_content[k][i] for k in self.encoded_content.keys()
            }
            if self.encoded_labels is not None:
                feature = InputFeatures(**inputs, label=self.encoded_labels[i])
            else:
                feature = InputFeatures(**inputs, label=None)
            self.features.append(feature)

    def __len__(self) -> int:
        return self.len_dset

    def __getitem__(self, i: int) -> InputFeatures:
        return self.features[i]

    def get_labels(self) -> T.List[str]:
        return self.labels


class ClassifierModel(object):
    """Trainable BERT-based classifier given a training & eval set."""

    def __init__(
        self,
        labels: T.List[str],
        model_path: str,
        cache_dir: str,
        output_dir: T.Optional[str] = None,
        num_train_epochs: T.Optional[int] = None,
        train_file: T.Optional[str] = None,
        dev_file: T.Optional[str] = None,
    ):
        """.

        Args:
            labels: list of valid labels used in the dataset
            model_path: name of model being used or filepath to where the model is stored
            model_path_tokenizer: name or path of tokenizer being used.
            cache_dir: directory where cache & output are kept.
            num_train_epochs: obvious. Why? To make unit testing faster.
            train_file: Required if we want to do .train()
            dev_file:: Required if we want to do .train_and_evaluate()

        """

        self.cache_dir = cache_dir
        self.model_path = model_path
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.num_splits = 5
        # self.num_train_epochs = 1
        # self.num_splits = 3

        self.labels = labels
        self.num_labels = len(labels)
        self.task_name = "classification"

        self.label_map = {label: i for i, label in enumerate(self.labels)}
        self.label_map_reverse = {i: label for i, label in enumerate(self.labels)}

        self.average_metrics = {}
        self.config = AutoConfig.from_pretrained(
            self.model_path,
            num_labels=self.num_labels,
            finetuning_task=self.task_name,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, cache_dir=self.cache_dir,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            from_tf=False,
            config=self.config,
            cache_dir=self.cache_dir,
        )

        if train_file is not None:
            self.train_dataset = self.make_dataset(
                train_file, Settings.CONTENT_COL, Settings.LABEL_COL,
            )
            self.train_dataset_path = train_file
        else:
            self.train_dataset = None
        if dev_file is not None:
            self.eval_dataset = self.make_dataset(
                dev_file, Settings.CONTENT_COL, Settings.LABEL_COL,
            )
        else:
            self.eval_dataset = None

    def compute_metrics(self, p: EvalPrediction) -> ClassifierMetricsJson:
        """
        Compute accuracy of predictions vs labels. Piggy back on sklearn.
        """
        y_pred_nominal = np.argmax(p.predictions, axis=1)
        y_pred = [self.labels[i] for i in y_pred_nominal]
        y_true_nominal = p.label_ids
        y_true = [self.labels[i] for i in y_true_nominal]

        clsf_report_sklearn = classification_report(
            y_true=y_true, y_pred=y_pred, output_dict=True, labels=self.labels,
        )
        final = ClassifierMetricsJson(
            {
                "accuracy": clsf_report_sklearn["accuracy"],
                "macro_f1_score": clsf_report_sklearn["macro avg"]["f1-score"],
                "macro_recall": clsf_report_sklearn["macro avg"]["recall"],
                "macro_precision": clsf_report_sklearn["macro avg"]["precision"],
            }
        )
        return final

    def make_dataset(
        self, fname: str, content_column: str, label_column: T.Optional[str]
    ) -> ClassificationDataset:
        """Create a Torch dataset object from a file using the built-in tokenizer.

        Inputs:
            fname: name of the file being used
            content_column: column that contains the text we want to analyze
            label_column: column containing the label

        Returns:
            ClassificationDataset object (which is a Torch dataset underneath)
        """
        return ClassificationDataset(
            self.labels,
            self.tokenizer,
            self.label_map,
            fname,
            content_column,
            label_column,
        )
    
    def set_metrics(self, metric) -> None:
        for key in metric:
            if key not in self.average_metrics:
                self.average_metrics[key] = 0
            self.average_metrics[key] += metric[key] / self.num_splits

    def train(self) -> None:
        """Train a BERT-based model, using the training set to train & the eval set as
        validation.
        """
        assert self.train_dataset is not None, "train_file was not provided!"

        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                do_train=True,
                do_eval=True,
                evaluate_during_training=True,
                output_dir=self.output_dir,
                overwrite_output_dir=True,
                num_train_epochs=self.num_train_epochs,
            ),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        self.trainer.train(model_path=self.model_path)
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.trainer.args.output_dir)
    
    def train_no_evaluate(self) -> None:
        """Train a BERT-based model, using the training set to train.
        """
        assert self.train_dataset is not None, "train_file was not provided!"

        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                do_train=True,
                output_dir=self.output_dir,
                overwrite_output_dir=True,
                num_train_epochs=self.num_train_epochs,
            ),
            train_dataset=self.train_dataset,
        )
        self.trainer.train(model_path=self.model_path)
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.trainer.args.output_dir)

    def train_and_evaluate(self) -> ClassifierMetricsJson:
        """
        Wrapper on the trainer.evaluate method; evaluate model's performance on eval set
        provided by the user.
        """
        assert self.eval_dataset is not None, "dev_file was not provided!"
        assert self.train_dataset is not None, "train_file was not provided!"

        self.train()
        metrics: T.Dict[str, float] = self.trainer.evaluate(  # type: ignore[assignment]
            eval_dataset=self.eval_dataset
        )

        new_metrics: T.Dict[str, float] = {}
        for key, val in metrics.items():
            if key == "eval_loss":
                continue
            if "eval_" in key:  # transformers library prepends this
                new_key = key.replace("eval_", "")
                new_metrics[new_key] = val

        return ClassifierMetricsJson(
            {
                "accuracy": new_metrics["accuracy"],
                "macro_f1_score": new_metrics["macro_f1_score"],
                "macro_recall": new_metrics["macro_recall"],
                "macro_precision": new_metrics["macro_precision"],
            }
        )

    def do_cross_validation(self):
        """
        Performs k-fold cross validation on the data provided by the user
        """
        assert self.train_dataset_path is not None # train file not provided

        train_dset_df = pd.read_csv(self.train_dataset_path)
        train_dset_headers = train_dset_df.columns
        train_dset_table = train_dset_df.to_numpy() 

        ss = StratifiedKFold(n_splits=self.num_splits, shuffle=True)
        X, y = zip(*train_dset_table)
        split_counter = 0

        for train_indices, dev_indices in ss.split(X, y):
            app.logger.info('split '+ str(split_counter + 1)+ ' out of '+ str(self.num_splits)+ '-----')

            data_train = pd.DataFrame([train_dset_table[i] for i in train_indices], 
                columns = train_dset_headers)
            data_dev = pd.DataFrame([train_dset_table[i] for i in dev_indices],
                columns = train_dset_headers)
            
            train_tempfile_path = tempfile.mkstemp(suffix='.csv', prefix='kfold')
            dev_tempfile_path = tempfile.mkstemp(suffix='.csv', prefix='kfolddev')
            data_train.to_csv(train_tempfile_path[1])
            data_dev.to_csv(dev_tempfile_path[1])

            self.train_dataset = self.make_dataset(train_tempfile_path[1], 
                Settings.CONTENT_COL, Settings.LABEL_COL,)
            self.eval_dataset = self.make_dataset(dev_tempfile_path[1],
                Settings.CONTENT_COL, Settings.LABEL_COL,)

            metrics = self.train_and_evaluate() #training is done here.
            self.set_metrics(metrics)
            
            os.remove(train_tempfile_path[1])
            os.remove(dev_tempfile_path[1])
            
            split_counter += 1
        return self.average_metrics

    def perform_cv_and_train(self):
        print('hererer')
        new_metrics = self.do_cross_validation()
        print('done with cv preparing for training with whole dataset...')
        self.train_dataset = self.make_dataset(
                self.train_dataset_path, Settings.CONTENT_COL, Settings.LABEL_COL,
            )
        self.train_no_evaluate()
        return ClassifierMetricsJson(
            {
                "accuracy": new_metrics["accuracy"],
                "macro_f1_score": new_metrics["macro_f1_score"],
                "macro_recall": new_metrics["macro_recall"],
                "macro_precision": new_metrics["macro_precision"],
            }
        )

    def predict_and_save_predictions(
        self,
        test_set_path: str,
        content_column: str,
        predicted_column: str,
        output_file_path: str,
    ) -> None:
        """
        Given a path to a dataset and the column containing text, 
        provide the labels predicted by the model.

        Inputs:
            test_set_path: absolute filepath of inference dataset
            content_column: column containing the text we'll analyze
            predicted_column: what to name the column with predictions.
            output_file_path: path where the CSV of predictions.

        WritesTo:
            output_file_path: A CSV with two columns: examples, and predictions.
        """
        # The transformers library should not require an output_dir if we're only
        # predicting using the model and getting the results returned in Python.
        # But alas, it does.
        if self.output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="abc_prediction")
        else:
            output_dir = self.output_dir

        test_dset = self.make_dataset(test_set_path, content_column, None)
        trainer: Trainer[None] = Trainer(
            model=self.model,
            args=TrainingArguments(
                do_train=False,
                do_eval=False,
                do_predict=False,
                output_dir=output_dir,
                num_train_epochs=self.num_train_epochs,
            ),
        )

        pred_output: PredictionOutput = trainer.predict(test_dset)

        y_pred = pred_output.predictions.argmax(axis=1)
        preds_in_user_labels = [test_dset.labels[i] for i in y_pred]

        pred_series = pd.Series(preds_in_user_labels, name=predicted_column)  # type: ignore
        output_df: pd.DataFrame = pd.concat(
            [test_dset.content_series, pred_series],  # type: ignore
            axis=1,
            names=[test_dset.content_series, pred_series.name],  # type: ignore
        )

        output_df.to_csv(output_file_path, index=False)
