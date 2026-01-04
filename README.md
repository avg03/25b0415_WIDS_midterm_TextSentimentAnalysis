 #Week 1: Python Fundamentals & Data Manipulation
This notebook focuses on building a strong foundation in Python libraries essential for Data Science.

Key Concepts Covered:
Python Basics:

Differences between Lists and Dictionaries.

Input handling, string formatting, and basic function definitions.

List operations (append, remove, loops).

NumPy (Numerical Python):

Array initialization (zeros, ones, random).

Understanding array shapes and dimensions.

Advanced slicing (1D, 2D) and accessing subsets of data.

Pandas:

Creating DataFrames from lists and dictionaries.

Indexing & Slicing: Detailed comparison and implementation of .loc (label-based) vs .iloc (integer-based).

Subset extraction based on specific rows and columns.

#Week 2: Natural Language Processing (NLP)
This notebook implements a Sentiment Analysis pipeline using the Corona_NLP_test.csv dataset. The assignment explores two different approaches to text classification.

1. Data Preprocessing
Before modeling, raw tweets were cleaned using a custom processing pipeline:

Cleaning: Removal of HTML tags, URLs, and punctuation using Regex.

Emoji Handling: Removing emojis using the emoji library.

Normalization: Spell correction using TextBlob and stopword removal using NLTK.

2. System A: Statistical Approach (TF-IDF)
Feature Extraction: Used TfidfVectorizer (Term Frequency-Inverse Document Frequency) with a max feature limit.

Model: Logistic Regression (sklearn).

Performance: Achieved high training accuracy (~93%) on the split data.

3. System B: Embedding & Neural Network Approach
Tokenization: Utilized nltk.word_tokenize.

Embeddings: Trained a Word2Vec model using Gensim to create dense vector representations of words. Document vectors were created by averaging word vectors.

Model: A Sequential Neural Network using TensorFlow/Keras:

Hidden Layers: Dense layers with ReLU activation.

Regularization: Dropout layer (0.2) to prevent overfitting.

Output Layer: Softmax activation for multi-class classification.

Optimization: Adam optimizer with Sparse Categorical Crossentropy loss.

4. Theoretical Understanding
Addressed the importance of Tokenization and Embeddings in Transformer architectures, explaining how models process numerical representations rather than raw text.

🛠️ Libraries & Dependencies
To run these notebooks, the following Python libraries are required:

Core: numpy, pandas

NLP & Text Processing: nltk, textblob, emoji, gensim

Machine Learning: scikit-learn

Deep Learning: tensorflow (Keras)