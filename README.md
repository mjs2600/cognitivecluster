# Cognitive Cluster

[![Build Status](https://travis-ci.org/mjs2600/cognitivecluster.svg?branch=master)](https://travis-ci.org/mjs2600/cognitivecluster)

Cognitive Cluster is a library for clustering written metacognitive exercises based on document embeddings.
This is useful for creating cross-cluster partnerships to increase the diversity of problem solving techniques in teams.

## Installation

To instal `cognitivecluster`, run `pip install cognitivecluster`. The library currently only supports Python 3.6. This is a limitation in TensorFlow, as soon as Python 3.7 support is released, I will update this library to add 3.7 support.

## Usage

To cluster people based on writing samples, pass either a Numpy array or a Pandas series to `cognitivecluster.sentence_vector_clusters`.

### Examples

```python
>>> import cognitivecluster
>>> cognitivecluster.sentence_vector_clusters(np_sentence_array)
array([0, 0, 1, 1, 2, 2], dtype=int32)
>>> df['cluster'] = cognitivecluster.sentence_vector_clusters(df.metacognitive_exercises)
>>> df
  metacognitive_exercises  cluster
0          I like carrots        0
1         I like potatoes        0
2              I am a cat        1
3              I am a dog        1
4          This is a test        2
5      This is a sequence        2
```
