"""LDA processing."""
import re
import string
import typing as T
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import typing_extensions as TT
from gensim import corpora  # type: ignore
from gensim import models
from gensim.models import CoherenceModel  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.stem.wordnet import WordNetLemmatizer  # type: ignore

from flask import current_app as app
from flask_app.settings import Settings
# from modeling.settings2 import Settings
EXCEL_EXTENSIONS = {"xlsx", "xls"}
CSV_EXTENSIONS = {"csv"}
TSV_EXTENSIONS = {"tsv"}


EXPERT_LABEL_COLUMN_NAME = "EXPERT_LABEL"

# import logging

# logger = logging.getLogger(__name__)


class TopicModelMetricsJson(TT.TypedDict):
    umass_coherence: float


class LDAPreprocessingOptions(TT.TypedDict, total=False):
    remove_phrases: bool
    join_phrases: bool
    remove_punctuation_and_digits: bool
    remove_stopwords: bool
    lemmatize_content: bool
    remove_short_words: bool


class Corpus(object):
    """Creates a dataset suitable for LDA analysis; does text preprocessing."""

    def __init__(
        self,
        file_name: str,
        content_column_name: str,
        id_column_name: str,
        language: str = "english",
        min_word_length: int = 2,
        extra_punctuation: T.Set[str] = set(),
        extra_stopwords: T.List[str] = [],
        phrases_to_join: T.List[str] = [],
        phrases_to_remove: T.List[str] = [],
        dont_stem: T.Set[str] = set(),
        processing_to_do: LDAPreprocessingOptions = LDAPreprocessingOptions({}),
    ):
        """
        Args:
            file_name: The CSV or Excel file.
            content_column_name: The header of content column.
            id_column_name:  The header of the id column.
            language: The language to use for stopwords.
            min_word_length: Minimum length of a word after stemming.
            extra_punctuation: Additional punctuation to be removed.
            extra_stopwords: Stopwords additional to the list obtained from ntlk using 
                `language` arg above.
            phrases_to_join: Phrases that would normally be split into multiple words, 
                but should be retained as a single word.
            phrases_to_remove: Phrases to remove.
            dont_stem: Words that should not be stemmbed.
            processing_to_do: What preprocessing to do. Look at LDAPreprocessingOptions 
                for the available options. If any preprocessing option is omitted, (as 
                opposed to being set to False explicitly), that preprocessing option
                will be executed.
            """
        file_path = Path(file_name)

        if not file_path.exists():
            raise ValueError("File {} doesn't exist!".format(file_name))

        suffix = file_path.suffix.strip(".")

        if suffix in CSV_EXTENSIONS:

            def doc_reader(b: str) -> pd.DataFrame:
                # dtype=object: Disable converting to non text column
                return pd.read_csv(b, dtype=object, na_filter=False)

        else:
            raise ValueError(
                "File type of {} is inconsistent or invalid!".format(file_name)
            )

        self.df_docs = doc_reader(file_name)
        self.id_column_name = id_column_name
        self.content_column_name = content_column_name
        
        self.processing_to_do = processing_to_do
        self.phrases_to_join = phrases_to_join
        # if(len(self.phrases_to_join)>0): self.processing_to_do['join_phrases'] =  True
        self.language = language
        self.phrases_to_remove = phrases_to_remove
        # if(len(self.phrases_to_remove)>0): self.processing_to_do['remove_phrases'] = True
        self.dont_stem = dont_stem
        self.min_word_length = min_word_length
        # self.processing_to_do['join_phrases'] = True
        punctuation_no_underscore = set(string.punctuation)
        punctuation_no_underscore.add("’")
        punctuation_no_underscore.add("”")
        punctuation_no_underscore.remove("_")
        self.punctuation = punctuation_no_underscore | extra_punctuation
        # self.processing_to_do['remove_punctuation_and_digits'] = True
        self.df_docs[Settings.STEMMED_CONTENT_COL] = self.df_docs[
            self.content_column_name
        ].apply(lambda b: b.lower())
        if(self.processing_to_do['remove_stopwords']):
            try:
                self.stopwords = stopwords.words(self.language)
            except OSError:
                raise ValueError(
                    "No stopwords exist for language {}!".format(self.language)
                )
        else:
            self.stopwords = []
        # if(language=='english'):
        #     self.processing_to_do.get('lemmatize_content', True)
        #     self.processing_to_do.get('remove_short_words', True)
        self.stopwords += extra_stopwords
        preprocessing_completed = []
        print(self.processing_to_do)
        # Figure out if the user meant to remove phrases by checking
        # if they provided phrases to remove
        remove_phrases_default = (phrases_to_remove != [])
        if remove_phrases_default:
            self.remove_phrases()
            preprocessing_completed.append("Removed phrases")

        if self.processing_to_do.get("join_phrases"):
            self.join_phrases()
            preprocessing_completed.append("Joined phrases")

        if self.processing_to_do.get("remove_punctuation_and_digits"):
            self.remove_punctuation_and_digits_and_tokenize()
            preprocessing_completed.append("Removed punctuation and digits")
        else:
            self.tokenize_content()
            preprocessing_completed.append("Tokenized content")
    
        if self.processing_to_do.get("remove_stopwords", True):
            self.remove_stopwords()
            preprocessing_completed.append("Removed stopwords")

        if self.processing_to_do.get("lemmatize_content", True):
            lemmatizer = WordNetLemmatizer()
            self.lemmatize_content(lemmatizer)
            preprocessing_completed.append("Lemmatized content")

        if self.processing_to_do.get("remove_short_words", True):
            self.remove_short_words(self.min_word_length)
            preprocessing_completed.append("Removed short words")

        self.preprocessing_completed = preprocessing_completed
        print(self.preprocessing_completed)

    def what_preprocessing_was_completed(self) -> T.List[str]:
        return self.preprocessing_completed

    def remove_phrases(self) -> bool:
        if len(self.phrases_to_remove) == 0:
            logger.warn("Asked to remove phrases, but self.phrases_to_remove is empty.")
            return True

        def remove_phrases_from_content(content: str) -> str:
            for w in self.phrases_to_remove:
                if w in content:
                    content = re.sub(w, "", content)

            return content

        self.df_docs[Settings.STEMMED_CONTENT_COL] = self.df_docs[
            Settings.STEMMED_CONTENT_COL
        ].apply(remove_phrases_from_content)

        return True

    def join_phrases(self) -> bool:
        def join_phrases_in_content(content: str) -> str:
            for w in self.phrases_to_join:
                if w in content:
                    content = re.sub(w, "_".join(w.split()), content)
            return content

        self.df_docs[Settings.STEMMED_CONTENT_COL] = self.df_docs[
            Settings.STEMMED_CONTENT_COL
        ].apply(join_phrases_in_content)

        return True

    def remove_punctuation_and_digits_and_tokenize(self) -> bool:
        def remove_punctuation_and_digits_from_content_and_tokenize(
            content: str,
        ) -> T.List[str]:
            content = content.translate(
                str.maketrans("", "", "".join(self.punctuation) + string.digits)
            )
            content_ls = [
                word for word in content.split() if re.match("[a-zA-Z0-9]+", word)
            ]
            return content_ls

        self.df_docs[Settings.STEMMED_CONTENT_COL] = self.df_docs[
            Settings.STEMMED_CONTENT_COL
        ].apply(remove_punctuation_and_digits_from_content_and_tokenize)

        return True

    def tokenize_content(self) -> bool:
        self.df_docs[Settings.STEMMED_CONTENT_COL] = self.df_docs[
            Settings.STEMMED_CONTENT_COL
        ].apply(lambda b: [w for w in b.split()])

        return True

    def remove_stopwords(self) -> bool:
        self.df_docs[Settings.STEMMED_CONTENT_COL] = self.df_docs[
            Settings.STEMMED_CONTENT_COL
        ].apply(lambda content: [c for c in content if c not in self.stopwords])

        return True

    def lemmatize_content(self, lemmatizer: WordNetLemmatizer) -> bool:
        self.df_docs[Settings.STEMMED_CONTENT_COL] = self.df_docs[
            Settings.STEMMED_CONTENT_COL
        ].apply(
            lambda content: [
                lemmatizer.lemmatize(c) for c in content if c not in self.dont_stem
            ]
        )

        return True

    def remove_short_words(self, min_length: int) -> bool:
        self.df_docs[Settings.STEMMED_CONTENT_COL] = self.df_docs[
            Settings.STEMMED_CONTENT_COL
        ].apply(lambda content: [c for c in content if len(c) > self.min_word_length])

        return True

    def random_sample(
        self,
        sample_size: int,
        spreadsheet_path: str,
        extra_df_columns_wanted: T.List[str] = [],
    ) -> bool:
        sample_df = self.df_docs.sample(n=sample_size)
        sample_df = sample_df[
            [self.id_column_name, "content_original"] + extra_df_columns_wanted
        ]
        sample_df[EXPERT_LABEL_COLUMN_NAME] = np.nan

        sample_df_writer = pd.ExcelWriter(spreadsheet_path)  # type: ignore
        sample_df.to_excel(sample_df_writer)
        sample_df_writer.save()

        return True


class LDAModeler(object):
    """Runs LDA modeling given a Corpus."""

    def __init__(
        self,
        content: Corpus,
        mallet_bin_directory: str,
        low_bound_dict: float = 0.02,
        high_bound_dict: float = 0.5,
        iterations: int = 1000,
    ):
        self.content = content
        self.mallet_bin_directory = mallet_bin_directory
        self.my_corpus = list(self.content.df_docs[Settings.STEMMED_CONTENT_COL])

        self.dictionary = corpora.Dictionary(self.my_corpus)
        self.dictionary.filter_extremes(
            no_below=low_bound_dict, no_above=high_bound_dict
        )
        self.corpus_bow = [self.dictionary.doc2bow(text) for text in self.my_corpus]

        self.lda_model: T.Optional[models.wrappers.LdaMallet] = None
        self.num_topics = 0
        self.iterations = iterations

    def model_topics(
        self,
        num_topics: int = 10,
        num_keywords: int = Settings.DEFAULT_NUM_KEYWORDS_TO_GENERATE,
    ) -> T.Tuple[
        float, T.List[T.List[str]], T.Any, T.Iterator[T.List[T.Tuple[int, float]]]
    ]:
        self.num_topics = num_topics
        mallet_path = Path(self.mallet_bin_directory) / "mallet"
        if not mallet_path.exists():
            raise Exception(
                f"Could not find a file named 'mallet' {str(self.mallet_bin_directory)}. Are"
                " you sure you installed Mallet and set MALLET_BIN_DIRECTORY correctly?"
            )
        self.lda_model = models.wrappers.LdaMallet( 
            str(mallet_path),
            corpus=self.corpus_bow,
            num_topics=num_topics,
            optimize_interval=10,
            id2word=self.dictionary,
            random_seed=1,
            iterations=self.iterations,
        )

        coherence_model = CoherenceModel(
            model=self.lda_model, corpus=self.corpus_bow, coherence="u_mass"
        )
        umass_coherence: np.float64 = coherence_model.get_coherence()

        topic_keywords: T.List[T.List[str]] = []
        for idx, topic in self.lda_model.show_topics(
            num_topics=num_topics, num_words=num_keywords, formatted=False
        ):
            topic_keywords.append([w[0] for w in topic])

        topic_proportions = self.get_topic_proportions()
        topics_by_doc: T.Iterator[
            T.List[T.Tuple[int, float]]
        ] = self.lda_model.load_document_topics()

        return float(umass_coherence), topic_keywords, topic_proportions, topics_by_doc

    def model_topics_to_spreadsheet(
        self,
        num_topics: int,
        default_topic_names: T.List[str],
        num_keywords: int = 20,
        fname_keywords: str = "topic_keywords_and_proportions.xlsx",
        fname_topics_by_doc: str = "topic_probabilities_by_document.xlsx",
        extra_df_columns_wanted: T.List[str] = [],
    ) -> TopicModelMetricsJson:
        """

        Returns:
            umass_coherence:
        """

        self.num_topics = num_topics
        (
            umass_coherence,
            topic_keywords,
            topic_proportions,
            topics_by_doc,
        ) = self.model_topics(self.num_topics, num_keywords)

        topic_keywords_df = pd.DataFrame()
        for w_idx in range(num_keywords):
            topic_keywords_df["word_{}".format(str(w_idx))] = [
                topic_keywords[i][w_idx] for i in range(len(topic_keywords))
            ]
        topic_keywords_df[Settings.TOPIC_PROPORTIONS_ROW] = topic_proportions
        assert len(topic_keywords_df.index) == num_topics
        topic_keywords_df.index = pd.Index(
            range(num_topics)
        )  # For now, just keep the topic indices, we'll do naming in the backend/(not modeling code)

        topic_keywords_df.T.to_csv(fname_keywords, index=True)  # type: ignore

        doc_topics = np.matrix(
            [[m[1] for m in mat] for mat in topics_by_doc]
        )  # create document --> topic proba matrix
        doc_topic_df = self.content.df_docs[
            [self.content.id_column_name]
            + extra_df_columns_wanted
            + [self.content.content_column_name]
            + [Settings.STEMMED_CONTENT_COL]
        ]

        for topic_num in range(self.num_topics):
            doc_topic_df[
                Settings.PROBAB_OF_TOPIC_TEMPLATE.format(default_topic_names[topic_num])
            ] = doc_topics[:, topic_num]
        doc_topic_df.set_index(self.content.id_column_name, inplace=True)

        # Again, just keep the indices of the topic indices here (as opposed to doing
        # something like "topic_0") because we'll be naming them in app.py.
        most_likely_topic_per_doc = [np.argmax(r) for r in doc_topics]

        doc_topic_df[Settings.MOST_LIKELY_TOPIC_COL] = most_likely_topic_per_doc
        doc_topic_df.to_csv(fname_topics_by_doc, index=True)

        return TopicModelMetricsJson(umass_coherence=float(umass_coherence))

    def get_topic_proportions(self) -> np.ndarray:  # type: ignore
        """
        Given a corpus and a model and a number of topics, get the topic probability
        distribution for each document in the corpus and use it to get the average
        topic proportions in that corpus for the model
        """
        if self.num_topics == 0 or self.lda_model is None:
            raise ValueError(
                "Number of topics not assigned and/or LDA model not trained!"
            )

        group_topic_proba = np.zeros(self.num_topics)
        topics = self.lda_model[self.corpus_bow]
        for td in topics:
            group_topic_proba = group_topic_proba + np.array([t[1] for t in td])
        z = group_topic_proba / sum(group_topic_proba)

        return z
