import logging
import typing as T

import typing_extensions as TT
from redis import Redis
from rq import Queue  # type: ignore

from flask_app.settings import Settings
from flask import current_app as app
# logging.basicConfig()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class ClassifierPredictionTaskArgs(TT.TypedDict):
    task_type: TT.Literal["prediction"]  # This redundancy is for mypy's sake
    test_set_id: int
    test_file: str
    labels: T.List[str]
    model_path: str
    cache_dir: str
    test_output_file: str


class ClassifierTrainingTaskArgs(TT.TypedDict):
    task_type: TT.Literal["training"]
    classifier_id: int
    labels: T.List[str]
    model_path: str
    cache_dir: str
    num_train_epochs: float
    train_file: str
    dev_file: str
    output_dir: str


class TopicModelTrainingTaskArgs(TT.TypedDict):
    # No need for task type, since the topic model queue has only one kind of task
    topic_model_id: int
    training_file: str
    fname_keywords: str
    fname_topics_by_doc: str
    iterations: int
    mallet_bin_directory: str
    language: str

class TopicModelProcessingOptions(TT.TypedDict):
    remove_stopwords: bool
    extra_stopwords: T.List[str]
    phrases_to_join: T.List[str]
    remove_punctuation: bool
    do_stemming: bool
    do_lemmatizing: bool
    min_word_length: int

class QueueManager(object):
    def __init__(self) -> None:
        connection = Redis(host=Settings.REDIS_HOST, port=Settings.REDIS_PORT)
        is_async = True
        self.classifiers_queue = Queue(
            name="classifiers", connection=connection, is_async=is_async
        )
        self.topic_models_queue = Queue(
            name="topic_models", connection=connection, is_async=is_async
        )

    def add_classifier_training(
        self,
        classifier_id: int,
        labels: T.List[str],
        model_path: str,
        train_file: str,
        dev_file: str,
        cache_dir: str,
        output_dir: str,
        num_train_epochs: int = 3,
    ) -> None:
        app.logger.info("Enqueued classifier training")
        app.logger.info('hererer')
        self.classifiers_queue.enqueue(
            "flask_app.modeling.tasks.do_classifier_related_task",
            ClassifierTrainingTaskArgs(
                task_type="training",
                classifier_id=classifier_id,
                num_train_epochs=num_train_epochs,
                labels=labels,
                model_path=model_path,
                train_file=train_file,
                dev_file=dev_file,
                cache_dir=cache_dir,
                output_dir=output_dir,
            ),
            job_timeout=-1,  # This will be popped off by RQ
        )

    def add_classifier_prediction(
        self,
        test_set_id: int,
        labels: T.List[str],
        model_path: str,
        test_file: str,
        cache_dir: str,
        test_output_file: str,
    ) -> None:

        app.logger.info("Enqueued classifier training.")
        self.classifiers_queue.enqueue(
            "flask_app.modeling.tasks.do_classifier_related_task",
            ClassifierPredictionTaskArgs(
                test_set_id=test_set_id,
                task_type="prediction",
                labels=labels,
                model_path=model_path,
                test_file=test_file,
                cache_dir=cache_dir,
                test_output_file=test_output_file,
            ),
            job_timeout=-1,
        )

    def add_topic_model_training(
        self,
        topic_model_id: int,
        training_file: str,
        fname_keywords: str,
        fname_topics_by_doc: str,
        mallet_bin_directory: str,
        language: str,
        remove_stopwords: bool,
        extra_stopwords: list,
        phrases_to_join: list,
        remove_punctuation: bool,
        do_stemming: bool,
        do_lemmatizing: bool,
        min_word_length: int = 2,
        iterations: int = 1000
    ) -> None:
        app.logger.info("Enqueued lda training with pickle_data.")
        topic_mdl_processing_temp = TopicModelProcessingOptions(
            remove_stopwords=remove_stopwords,
            extra_stopwords=[] if extra_stopwords is None else extra_stopwords,
            phrases_to_join=[] if phrases_to_join is None else phrases_to_join,
            remove_punctuation=remove_punctuation,
            do_stemming=do_stemming,
            do_lemmatizing=do_lemmatizing,
            min_word_length=min_word_length
        )
        self.topic_models_queue.enqueue(
            "flask_app.modeling.tasks.do_topic_model_related_task",
            TopicModelTrainingTaskArgs(
                topic_model_id=topic_model_id,
                training_file=training_file,
                fname_keywords=fname_keywords,
                fname_topics_by_doc=fname_topics_by_doc,
                iterations=iterations,
                mallet_bin_directory=mallet_bin_directory,
                language=language
            ),
            topic_mdl_processing_temp
            ,
            job_timeout=-1,
        )
