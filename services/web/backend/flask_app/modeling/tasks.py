"""Tasks done in background by Rq.

This file must NOT be imported by app.py, or by any import thereof, because it will cause
a cyclic import.

The functions within are referred with qualified "Python paths"
(eg., "flask_app.modeling.tasks.do_classifier_related_task"). Rq supports that.
"""
import logging
import typing as T

from flask import url_for

import flask_app
from flask_app import emails
from flask_app.database import models
from flask_app.modeling.classifier import ClassifierModel
from flask_app.modeling.lda import Corpus
from flask_app.modeling.lda import LDAModeler
from flask_app.modeling.lda import LDAPreprocessingOptions
from flask_app.modeling.queue_manager import ClassifierPredictionTaskArgs
from flask_app.modeling.queue_manager import ClassifierTrainingTaskArgs
from flask_app.modeling.queue_manager import TopicModelTrainingTaskArgs, TopicModelProcessingOptions
from flask_app.settings import Settings
from flask import current_app as app
# logging.basicConfig()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


@flask_app.app.needs_app_context
def do_classifier_related_task(
    task_args: T.Union[ClassifierTrainingTaskArgs, ClassifierPredictionTaskArgs],
) -> None:
    if task_args["task_type"] == "prediction":
        test_set = models.TestSet.get(models.TestSet.id_ == task_args["test_set_id"])
        assert test_set.inference_began
        assert not test_set.inference_completed

        try:
            classifier_model = ClassifierModel(
                labels=task_args["labels"],
                model_path=task_args["model_path"],
                cache_dir=task_args["cache_dir"],
            )
            app.logger.info('reached here')
            classifier_model.predict_and_save_predictions(
                test_set_path=task_args["test_file"],
                content_column=Settings.CONTENT_COL,
                predicted_column=Settings.PREDICTED_LABEL_COL,
                output_file_path=task_args["test_output_file"],
            )
        except BaseException as e:
            app.logger.critical(f"Error while doing prediction task: {e}")
            test_set.error_encountered = True
        else:
            test_set.inference_completed = True
            emailer = emails.Emailer()
            emailer.send_email(
                email_template_name="classifier_inference_finished",
                to_email=test_set.notify_at_email,
                classifier_name=test_set.classifier.name,
                classifier_id=test_set.classifier.classifier_id,
                predictions_url=url_for(
                    "ClassifiersTestSetsPredictions",
                    classifier_id=test_set.classifier.classifier_id,
                    test_set_id=test_set.id_,
                    file_type=Settings.DEFAULT_FILE_FORMAT.strip("."),
                    _method="GET",
                ),
            )
        finally:
            test_set.save()

    elif task_args["task_type"] == "training":
        app.logger.info('hererer')
        assert task_args["task_type"] == "training"
        clsf = models.Classifier.get(
            models.Classifier.classifier_id == task_args["classifier_id"]
        )
        assert clsf.train_set is not None
        # assert clsf.dev_set is not None

        try:
            classifier_model = ClassifierModel(
                labels=task_args["labels"],
                num_train_epochs=task_args["num_train_epochs"],
                model_path=task_args["model_path"],
                train_file=task_args["train_file"],
                dev_file=task_args["dev_file"],
                cache_dir=task_args["cache_dir"],
                output_dir=task_args["output_dir"],
            )
            metrics = classifier_model.perform_cv_and_train()
        except BaseException as e:
            app.logger.critical(f"Error while doing classifier training task: {e}")
            clsf.train_set.error_encountered = True
            # clsf.dev_set.error_encountered = True
        else:
            clsf.train_set.training_or_inference_completed = True
            # clsf.dev_set.training_or_inference_completed = True
            clsf.train_set.metrics = models.ClassifierMetrics(**metrics)
            clsf.train_set.metrics.save()
            emailer = emails.Emailer()
            emailer.send_email(
                email_template_name="classifier_training_finished",
                to_email=clsf.notify_at_email,
                classifier_name=clsf.name,
                classifier_id=clsf.classifier_id,
                metrics=T.cast(T.Dict[str, float], metrics),
            )

        finally:
            # clsf.dev_set.save()
            clsf.train_set.save()
            clsf.save()


@flask_app.app.needs_app_context
def do_topic_model_related_task(task_args: TopicModelTrainingTaskArgs, 
                                processing_opts: TopicModelProcessingOptions
                            ) -> None:
    topic_mdl = models.TopicModel.get(
        models.TopicModel.id_ == task_args["topic_model_id"]
    )
    assert topic_mdl.lda_set is not None
    try:
        preprocessing_opts = LDAPreprocessingOptions(
            remove_phrases=False,
            join_phrases=True if len(processing_opts['phrases_to_join'])>0 else False,
            remove_punctuation_and_digits = processing_opts['remove_punctuation'],
            remove_stopwords=processing_opts['remove_stopwords'],
            lemmatize_content=processing_opts['do_lemmatizing'] if task_args['language']=='english' else False,
        ) # nothing being done about 'do_stemming'
        corpus = Corpus(
            file_name=task_args["training_file"],
            content_column_name=Settings.CONTENT_COL,
            id_column_name=Settings.ID_COL,
            language=task_args["language"],
            extra_stopwords=processing_opts['extra_stopwords'],
            phrases_to_join=processing_opts['phrases_to_join'],
            min_word_length=processing_opts['min_word_length'],
            processing_to_do=preprocessing_opts
        )
        lda_modeler = LDAModeler(
            corpus,
            iterations=task_args["iterations"],
            mallet_bin_directory=task_args["mallet_bin_directory"],
        )
    except Exception as e:
        app.logger.critical(f"Error while doing lda training task: {e}")

        topic_mdl.lda_set.error_encountered = True
    else:
        metrics = lda_modeler.model_topics_to_spreadsheet(
            num_topics=topic_mdl.num_topics,
            default_topic_names=topic_mdl.topic_names,
            fname_keywords=task_args["fname_keywords"],
            fname_topics_by_doc=task_args["fname_topics_by_doc"],
        )
        topic_mdl.lda_set.metrics = models.TopicModelMetrics.create(**metrics)
        topic_mdl.lda_set.lda_completed = True
        emailer = emails.Emailer()
        emailer.send_email(
            email_template_name="topic_model_training_finished",
            to_email=topic_mdl.notify_at_email,
            topic_model_name=topic_mdl.name,
            topic_model_id=topic_mdl.id_,
            topic_model_preview_url=f"http://{Settings.SERVER_NAME}/playground.html?step=1&id={topic_mdl.id_}",
            metrics=T.cast(T.Dict[str, T.Union[int, float]], metrics),
        )

    finally:
        topic_mdl.lda_set.save()
