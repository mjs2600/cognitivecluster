import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from tensorflow import Session
from tensorflow_hub import Module

from cognitivecluster import __version__, sentence_vector_clusters, sentence_vectors


@pytest.mark.slow
def test_sentence_vector_clusters_integration():
    sentences = np.array(
        ["carrot", "tomato", "puppy", "dog", "metacognition", "problem solving"]
    )
    results = sentence_vector_clusters(sentences)
    assert len(results) == 6
    assert len(set(results)) == 3
    assert results[0] == results[1]
    assert results[2] == results[3]
    assert results[4] == results[5]


@pytest.mark.slow
def test_pandas_sentence_vector_clusters_integration():
    sentences = pd.Series(
        ["carrot", "tomato", "puppy", "dog", "metacognition", "problem solving"]
    )
    results = sentence_vector_clusters(sentences)
    assert len(results) == 6
    assert len(set(results.values)) == 3
    assert results.iloc[0] == results.iloc[1]
    assert results.iloc[2] == results.iloc[3]
    assert results.iloc[4] == results.iloc[5]


def test_sentence_vector_clusters(mocker):
    n_clusters = 3
    embeddings = np.array([[0], [0], [1], [1], [2], [2]])
    clusters = np.array([0, 0, 1, 1, 2, 2])
    sv_mock = mocker.patch("cognitivecluster.clusters.sentence_vectors")
    sv_mock.return_value = embeddings
    cluster_cls_mock = mocker.MagicMock(KMeans)
    cluster_mock = mocker.MagicMock(KMeans, autospec=True)
    cluster_cls_mock.return_value = cluster_mock
    cluster_mock.fit_predict.return_value = clusters
    sentences = np.array(
        ["carrot", "tomato", "puppy", "dog", "metacognition", "problem solving"]
    )
    results = sentence_vector_clusters(sentences, n_clusters, cluster_cls_mock)
    sv_mock.assert_called_once_with(sentences)
    cluster_cls_mock.assert_called_once_with(n_clusters=n_clusters)
    cluster_mock.fit_predict.assert_called_once_with(embeddings)
    assert (results == clusters).all()


def test_sentence_vectors(mocker):
    embeddings = np.array([[0], [0], [1], [1], [2], [2]])
    sentences = np.array(
        ["carrot", "tomato", "puppy", "dog", "metacognition", "problem solving"]
    )
    module_cls_mock = mocker.patch("tensorflow_hub.Module")
    module_mock = mocker.MagicMock(Module, autospec=True)
    tf_mock = mocker.patch("tensorflow.Session")
    session_mock = mocker.MagicMock(Session, autospec=True)
    session_mock.return_value = embeddings
    tf_mock.return_value.__enter__.return_value = session_mock
    module_cls_mock.return_value = module_mock
    module_mock.return_value = embeddings
    results = sentence_vectors(sentences)
    module_mock.assert_called_once_with(sentences)
    session_mock.run.assert_called_with(embeddings)
    assert (results == embeddings).all()
