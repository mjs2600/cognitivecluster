"""
A collection of metacognitive clustering techniques.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.cluster import KMeans


def sentence_vectors(sentences: np.array) -> np.ndarray:
    """
    `sentence_vectors` uses Google's universal sentence encoder to convert a
    Numpy array of natural language into a document embedding. 
    """
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embedding = session.run(embed(sentences))
    return embedding


def sentence_vector_clusters(
    sentences: np.array, n_clusters: int = 3, clusterer_cls=KMeans
) -> np.array:
    """
    `sentence_vector_clusters` takes a Numpy array of natural language and 
    clusters the elements.
    """
    pandas = False
    if hasattr(sentences, "values"):
        index = sentences.index
        sentences = sentences.values
        pandas = True
    clusterer = clusterer_cls(n_clusters=n_clusters)
    sv = sentence_vectors(sentences)
    clusters = clusterer.fit_predict(sv)
    if pandas:
        clusters = pd.Series(clusters, index=index)
    return clusters
