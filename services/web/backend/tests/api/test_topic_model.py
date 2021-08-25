import csv
import unittest
from unittest import mock

import pandas as pd  # type: ignore
from flask import current_app
from flask import url_for
from tests.common import AppMixin
from tests.common import make_csv_file
from tests.common import RQWorkerMixin

from flask_app import utils
from flask_app.app import API_URL_PREFIX
from flask_app.app import TopicModelStatusJson
from flask_app.database import models
from flask_app.modeling.queue_manager import QueueManager
from flask_app.settings import Settings
from flask_app.utils import Json


class TopicModelMixin(RQWorkerMixin, AppMixin):
    def setUp(self) -> None:
        super().setUp()
        num_topics = 10
        self._topic_mdl = models.TopicModel.create(
            name="test_topic_model",
            num_topics=num_topics,
            topic_names=[
                Settings.DEFAULT_TOPIC_NAME_TEMPLATE.format(topic_num)
                for topic_num in range(1, num_topics + 1)
            ],
            notify_at_email="davidat@bu.edu",
        )

        # Make sure the directory for the topic model exists
        utils.Files.topic_model_dir(self._topic_mdl.id_, ensure_exists=True)

        self._valid_training_table = [
            [cell]
            for cell in [
                # Taken from huffpost.com
                f"{Settings.CONTENT_COL}",
                "Florida Officer Who Was Filmed Shoving A Kneeling Black Protester Has Been Charged",
                "Fox News Host Ed Henry Fired After Sexual Misconduct Investigation",
                "Hong Kong Police Make First Arrests Under New Security Law Imposed By China",
                "As Democrats Unveil Ambitious Climate Goals, House Lawmakers Press For Green Stimulus",
                "Citing Racial Bias, San Francisco Will End Mug Shots Release",
                "‘Your Chewing Sounds Like Nails On A Chalkboard’: What Life With Misophonia Is Like",
                "Puerto Rico’s Troubled Utility Is A Goldmine For U.S. Contractors",
                "Schools Provide Stability For Refugees. COVID-19 Upended That.",
                "Jada Pinkett Smith Denies Claim Will Smith Gave Blessing To Alleged Affair",
                "College Students Test Positive For Coronavirus, Continue Going To Parties Anyway",
                "A TikTok User Noticed A Creepy Thing In ‘Glee’ You Can’t Unsee",
                "Prince Harry Speaks Out Against Institutional Racism: It Has ‘No Place’ In Society",
                "A Poet — Yes, A Poet — Makes History On ‘America’s Got Talent’",
                "I Ate At A Restaurant In What Was Once COVID-19’s Deadliest County",
                "This Is What Racial Trauma Does To The Body And Brain",
                "How To Avoid Bad Credit As Protections In The CARES Act Expire",
                "Here’s Proof We Need Better Mental Health Care For People Of Color",
                "“I hope that this is real,” Lauren Boebert said of the deep-state conspiracy theory.",
                "U.S. Buys Virtually All Of Coronavirus Drug Remdesivir In The World",
                "Trials found that the anti-viral drug can reduce the recovery time of COVID-19 patients by four days.",
                "Florida Gov. Ron DeSantis Says He Won’t Reinstate Restrictions Despite COVID-19 Surge",
                "Conservative Columnist Spells Out Exactly Who’s To Blame For U.S. Coronavirus Failings",
                "'We are living — and now dying — in an idiocracy of our own creation,' said The Washington Post's Max Boot.",
                "Lori Vallow Daybell allegedly conspired with her new husband to hide or destroy the bodies on his rural Idaho property.",
                "Obama Photographer Flags Yet Another Wild Difference With The Trump Presidency",
                "Viola Davis’ Call To ‘Pay Me What I’m Worth’ Is What The World Needs Now",
                "The Oscar winner's fierce declaration reemerged on social media to turbocharge calls for equity in Hollywood.",
                "Anderson Cooper Breaks Down Why Trump’s America Is Now ‘A Pariah State’",
            ]
        ]

        # What we expect to see after upload file being processed in the backend
        self._expected_training_table = [[Settings.ID_COL, Settings.CONTENT_COL]] + [
            [str(row_num), cell]
            for row_num, (cell,) in enumerate(self._valid_training_table[1:])
        ]


class TestTopicModels(TopicModelMixin, unittest.TestCase):
    def test_get(self) -> None:
        url = API_URL_PREFIX + "/topic_models/"
        expected_topic_model_json = TopicModelStatusJson(
            {
                "topic_model_id": self._topic_mdl.id_,
                "topic_model_name": "test_topic_model",
                "num_topics": self._topic_mdl.num_topics,
                "topic_names": [
                    Settings.DEFAULT_TOPIC_NAME_TEMPLATE.format(topic_num)
                    for topic_num in range(1, self._topic_mdl.num_topics + 1)
                ],
                "status": "not_begun",
                "metrics": None,
                "notify_at_email": self._topic_mdl.notify_at_email,
            }
        )
        with current_app.test_client() as client:
            with self.subTest("get all topic models"):
                with client.get(url) as resp:
                    self._assert_response_success(resp, url)
                    resp_json = resp.get_json()
                self.assertIsInstance(resp_json, list)
                expected_topic_model_list_json = [expected_topic_model_json]
                self.assertListEqual(resp_json, expected_topic_model_list_json)

            with self.subTest("get single topic model"):
                # Single entity endpoint
                single_topic_mdl_url = (
                    API_URL_PREFIX + f"/topic_models/{self._topic_mdl.id_}"
                )
                with client.get(single_topic_mdl_url) as single_topic_mdl_resp:
                    self._assert_response_success(
                        single_topic_mdl_resp, single_topic_mdl_url
                    )
                    single_topic_mdl_resp_json = single_topic_mdl_resp.get_json()
                self.assertIsInstance(single_topic_mdl_resp_json, dict)
                self.assertDictEqual(
                    single_topic_mdl_resp_json, dict(expected_topic_model_json)
                )

    def test_post(self) -> None:
        url = API_URL_PREFIX + "/topic_models/"

        num_topics = 2
        with current_app.test_client() as client:
            with self.subTest("creating a topic model"):
                with client.post(
                    url,
                    json={
                        "topic_model_name": "test_topic_model",
                        "num_topics": num_topics,
                        "notify_at_email": "davidat@bu.edu",
                    },
                ) as resp:
                    self._assert_response_success(resp, url)
                    resp_json: Json = resp.get_json()

                self.assertIsInstance(resp_json, dict)
                expected_topic_model_json = TopicModelStatusJson(
                    {
                        "topic_model_id": 999,
                        "topic_model_name": "test_topic_model",
                        "num_topics": 2,
                        "topic_names": [
                            Settings.DEFAULT_TOPIC_NAME_TEMPLATE.format(topic_num)
                            for topic_num in range(1, num_topics + 1)
                        ],
                        "status": "not_begun",
                        "metrics": None,
                        "notify_at_email": "davidat@bu.edu",
                    }
                )

                assert isinstance(resp_json, dict)
                resp_json.pop("topic_model_id")
                for key in expected_topic_model_json:
                    if key != "topic_model_id":
                        self.assertEqual(
                            expected_topic_model_json[key], resp_json[key]  # type: ignore[misc]
                        )


class TestTopicModelsTrainingFile(TopicModelMixin, unittest.TestCase):
    def test_post(self) -> None:
        # Mock the QueueManager
        queue_manager: QueueManager = current_app.queue_manager
        queue_manager.add_topic_model_training: mock.MagicMock = mock.MagicMock(return_value=None)  # type: ignore
        fname_keywords = utils.Files.topic_model_keywords_file(self._topic_mdl.id_)
        fname_topics_by_doc = utils.Files.topic_model_topics_by_doc_file(
            self._topic_mdl.id_
        )
        training_file_path = utils.Files.topic_model_training_file(self._topic_mdl.id_)

        test_url = API_URL_PREFIX + f"/topic_models/{self._topic_mdl.id_}/training/file"
        # Prepare the file to "upload"
        to_upload_file = make_csv_file(self._valid_training_table)

        with current_app.test_client() as client:
            res = client.post(test_url, data={"file": (to_upload_file, "train.csv")},)
            self._assert_response_success(res)

        # Assert that the correct training file was created in the correct directory
        self.assertTrue(training_file_path.exists())

        # Assert that the content of the training file matches
        self.maxDiff = 10000
        with training_file_path.open() as created_train_file:
            reader = csv.reader(created_train_file)
            created_training_table = list(reader)
        # The created training file should  have an ID column prepended
        self.assertSequenceEqual(created_training_table, self._expected_training_table)

        # Asssert the queue manager was called with the right arguments
        queue_manager.add_topic_model_training.assert_called_with(
            mallet_bin_directory=str(Settings.MALLET_BIN_DIRECTORY),
            topic_model_id=self._topic_mdl.id_,
            training_file=str(training_file_path),
            fname_keywords=str(fname_keywords),
            fname_topics_by_doc=str(fname_topics_by_doc),
        )

    def test_training(self) -> None:

        # Get some variables
        queue_manager: QueueManager = current_app.queue_manager
        training_file_path = utils.Files.topic_model_training_file(self._topic_mdl.id_)
        fname_keywords = utils.Files.topic_model_keywords_file(self._topic_mdl.id_)
        fname_topics_by_doc = utils.Files.topic_model_topics_by_doc_file(
            self._topic_mdl.id_
        )

        # Update db
        lda_set = models.LDASet()
        lda_set.save()
        self._topic_mdl.lda_set = lda_set
        self._topic_mdl.save()

        # Create the training file
        with training_file_path.open("w") as f:
            writer = csv.writer(f)
            writer.writerows(self._expected_training_table)

        # Start the training
        queue_manager.add_topic_model_training(
            mallet_bin_directory=str(Settings.MALLET_BIN_DIRECTORY),
            topic_model_id=self._topic_mdl.id_,
            training_file=str(training_file_path),
            fname_keywords=str(fname_keywords),
            fname_topics_by_doc=str(fname_topics_by_doc),
            iterations=10,
        )

        assert self._burst_workers("topic_models")

        with self.subTest("Test LDA file results are present"):
            self.assertTrue(fname_keywords.exists())
            self.assertTrue(fname_topics_by_doc.exists())

        with self.subTest("get request after training finished"):
            single_topic_mdl_url = (
                API_URL_PREFIX + f"/topic_models/{self._topic_mdl.id_}"
            )
            with current_app.test_client() as client, client.get(
                single_topic_mdl_url
            ) as resp:
                self._assert_response_success(resp)
                resp_json: Json = resp.get_json()
            assert isinstance(resp_json, dict)
            self.assertEqual(resp_json["status"], "completed")
            self.assertIsNotNone(resp_json["metrics"])
            self.assertLessEqual(  # presumably, the metrics are an integer or float
                set(map(type, resp_json["metrics"].values())), {int, float}
            )

        with self.subTest("topic previews"):
            url = API_URL_PREFIX + f"/topic_models/{self._topic_mdl.id_}/topics/preview"

            with current_app.test_client() as client, client.get(url) as resp:
                self._assert_response_success(resp)
                resp_json = resp.get_json()
            assert isinstance(resp_json, dict)
            self.assertIn("topic_previews", set(resp_json.keys()))
            self.assertEqual(
                len(resp_json["topic_previews"]), self._topic_mdl.num_topics
            )
            at_least_one_topic_has_examples = False
            for preview in resp_json["topic_previews"]:
                self.assertTrue("examples" in preview)
                self.assertTrue("keywords" in preview)

                # Check that the examples are longer than the keywords
                # doesnt NEED to be true, but should prbobably be true
                self.assertEqual(
                    len(preview["keywords"]), Settings.DEFAULT_NUM_KEYWORDS_TO_GENERATE
                )

                if len(preview["examples"]) > 0:
                    at_least_one_topic_has_examples = True
            # Test at least one topic has examples
            self.assertTrue(at_least_one_topic_has_examples)

        with self.subTest("download results file after naming topics"):
            topic_naming_url = url_for(
                "TopicModelsTopicsNames",
                id_=self._topic_mdl.id_,
                _method="POST",
                _external=False,
            )
            topic_names = [
                f"named_topic {i+1}" for i in range(self._topic_mdl.num_topics)
            ]
            post_data = {"topic_names": topic_names}
            with current_app.test_client() as client:
                with client.post(topic_naming_url, json=post_data) as resp:
                    self._assert_response_success(resp, url=topic_naming_url)
                    resp_json = resp.get_json()
            assert isinstance(resp_json, dict)
            self.assertListEqual(resp_json["topic_names"], topic_names)
            self.assertEqual(resp_json["status"], "completed")

        with self.subTest("get results of lda"):
            # Inspect the content of the keywords file
            expected_keywords_df_index = pd.Index(
                [f"word_{i}" for i in range(Settings.DEFAULT_NUM_KEYWORDS_TO_GENERATE)]
                + [Settings.TOPIC_PROPORTIONS_ROW]
            )
            expected_keywords_df_columns = pd.Index(topic_names)
            num_examples = len(self._valid_training_table) - 1  # -1 for the header

            expected_topics_by_doc_index = pd.Index(
                [f"{i}" for i in range(num_examples)], name="Id",
            )
            expected_topics_by_doc_columns = pd.Index(
                [Settings.CONTENT_COL, Settings.STEMMED_CONTENT_COL,]
                + [
                    Settings.PROBAB_OF_TOPIC_TEMPLATE.format(topic)
                    for topic in topic_names
                ]
                + [Settings.MOST_LIKELY_TOPIC_COL]
            )

            for file_type_with_dot in Settings.SUPPORTED_NON_CSV_FORMATS | {".csv"}:

                file_type = file_type_with_dot.strip(".")
                with self.subTest(
                    f"get topic modeling results: keywords: {file_type_with_dot}"
                ):
                    keywords_url = url_for(
                        "TopicModelsKeywords",
                        topic_model_id=self._topic_mdl.id_,
                        file_type=file_type,
                        _method="GET",
                    )

                    with current_app.test_client() as client, client.get(
                        keywords_url
                    ) as resp:
                        self._assert_response_success(resp, keywords_url)
                        keywords_df = self._df_from_bytes(
                            resp.data, file_type_with_dot, header=0, index_col=0  # type: ignore[arg-type]
                        )
                    keywords_df.index.name = None  # It doesn't matter
                    pd.testing.assert_index_equal(
                        keywords_df.index, expected_keywords_df_index
                    )

                    pd.testing.assert_index_equal(
                        keywords_df.columns, expected_keywords_df_columns
                    )
                with self.subTest(
                    f"get topic modeling results: topics_by_doc: {file_type_with_dot}"
                ):

                    topics_by_doc_url = url_for(
                        "TopicModelsTopicsByDoc",
                        topic_model_id=self._topic_mdl.id_,
                        file_type=file_type,
                        _method="GET",
                    )

                    with current_app.test_client() as client, client.get(
                        topics_by_doc_url
                    ) as resp:
                        self._assert_response_success(resp, topics_by_doc_url)
                        # Inspect the fname_topics_by_doc file
                        topics_by_doc_df = self._df_from_bytes(
                            resp.data, file_type_with_dot, header=0, index_col=0  # type: ignore[arg-type]
                        )

                    pd.testing.assert_index_equal(
                        topics_by_doc_df.index, expected_topics_by_doc_index
                    )

                    pd.testing.assert_index_equal(
                        topics_by_doc_df.columns, expected_topics_by_doc_columns,
                    )

    def test_error_during_training(self) -> None:

        # Get some variables
        queue_manager: QueueManager = current_app.queue_manager
        training_file_path = utils.Files.topic_model_training_file(self._topic_mdl.id_)
        fname_keywords = utils.Files.topic_model_keywords_file(self._topic_mdl.id_)
        fname_topics_by_doc = utils.Files.topic_model_topics_by_doc_file(
            self._topic_mdl.id_
        )

        # Mock db
        lda_set = models.LDASet()
        lda_set.save()
        self._topic_mdl.lda_set = lda_set
        self._topic_mdl.save()

        # NOTE: The exception will be raised here because we don't mock the training
        # file. Again, this is an error that can be checked for easily. But I don't know
        # how to raise an error in another process(see other NOTE in test_classifier.py)

        # Start the training
        queue_manager.add_topic_model_training(
            mallet_bin_directory=str(Settings.MALLET_BIN_DIRECTORY),
            topic_model_id=self._topic_mdl.id_,
            training_file=str(training_file_path),
            fname_keywords=str(fname_keywords),
            fname_topics_by_doc=str(fname_topics_by_doc),
            iterations=1,
        )

        self.assertTrue(self._burst_workers("topic_models"))
        self._topic_mdl = self._topic_mdl.refresh()
        self.assertEqual(self._topic_mdl.lda_set.error_encountered, True)  # type: ignore[union-attr]
        client = current_app.test_client()
        with client.get(
            url_for("OneTopicModel", topic_model_id=self._topic_mdl.id_)
        ) as resp:
            self._assert_response_success(resp)
            self.assertEqual(resp.get_json()["status"], "error_encountered")


if __name__ == "__main__":
    unittest.main()
