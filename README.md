# Introduction/ Features
TBC

# Project Structure
- AI-models/
    - BERT_hateful_comment_classification.ipynb
    - binary_classification_comment_dataset.ipynb
    - multilabel_classification_comment_dataset.ipynb
    - visualise_comment_dataset.ipynb
- dataset: contains cleaned and preprocessed dataset
- img/
- saved-models: contains saved models after training

# Machine Learning Models
TBC

## Simple Neural Network
![simple](./img/simple.png)
**Text Vectorization:** this layer will split a sentence into individual words and map each word to an interger in which computer can understand. For example: "I love cat" -> [23, 4, 67]

**Embedding:** this layer creates vector space of dimensions **embedding_dim**, and map the encoded word to a vector, capturing the semantic relationship between them.

**Global Max-pooling:** as the name inferred, the layer extracts the max value embedding vector. This reduces the dimenionality of output, ready to feed into next layer.

**Dense Layers:** fully connected neural networks that learn the complex patterns.

**Dropout Layers:** this layer dropout some neurons, preventing overfitting and forcing model learn more robust features.

## Convolutional Neural Network (CNN)
![cnn](./img/cnn.png)
**Text Vectorization & Embedding:** convert text into vector in multi-dimensional space.

**Convolutional Layers (Conv1D):** this layer apply filters that slide across the embedding vector, extracting local featuers and patterns.

**Max-pooling Layers:** this layer reduces the dimensionality of the features vectors. It retains the most prominent features from previous convolution layer, enforcing the learning process.

**Flatten Layer:** this layer flattens the multi-dimensional vector into a single-dimensional vector, ready to fed into dense layers.

**Dense Layers:** fully connected neural networks that learn the complex patterns.

## BERT model
![bert](./img/bert.png)

**BERT encoder:** this is a pre-trained language model that can understand context if text. This layer transforms the text into a numerical vector that captures the meaning and relationships between words.

**Dense Layers:** fully connected neural networks that learn the complex patterns.

*A word can have multiple meanings depending on the context. BERT embeddings represent words based on their context and surrounding words, while traditional word embeddings represent words independently of their context.*

# Outcome
This section presents the predictions of three distinct classification models.

| Binary Classification |  |
|---|---|
| Wish you cut your veins. Don't shout out you have mental problem | that's funny, I always pay them and I've never been to lunch |
| ![](./img/binary1.png) | ![](./img/binary2.png) |

| Multi-labeled Classification (CNN) |
|---|
| You idiot |
| ![](./img/multi-label-clf.png) |

| Multi-labeled Classification (BERT) |
|---|
| I hate myself |
| ![](./img/bert1.png) |
| I hate gay people |
| ![](./img/bert2.png) |