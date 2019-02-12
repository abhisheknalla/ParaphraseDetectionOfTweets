Goal

Given two tweets, predict whether they are paraphrases of each other.

Embeddings


    1. Obtain a set of features capturing the similarity between two tweets. Examples of these features are -
        TF-IDF based similarity between the two tweets
        Cosine distance between sentence vectors
        Word Mover's Distance between the two tweets
        Wordnet based similarity between nouns of the tweets
    2. Combine these features into a feature vector which represents the given pair of tweets.
    3. Classify this feature vector using Naive Bayes to obtain the similarity score between tweets.
    4. Classify using a Multi Layer Perceptron.

Recurrent Encoder Encode tweets using RNN based encoder followed by binary classification using softmax activation.
