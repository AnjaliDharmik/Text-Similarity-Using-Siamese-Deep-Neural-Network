# Text-Similarity-Using-Siamese-Deep-Neural-Network

Siamese neural network is a class of neural network architectures that contain two or more identical subnetworks. Identical means they have the same configuration with the same parameters and weights. Parameter updating is mirrored across both subnetworks. Instead of a model learning to classify its inputs, the neural networks learns to differentiate between two inputs. It learns the similarity between them.

It is a keras based implementation of Deep Siamese Bidirectional LSTM network to capture phrase/sentence similarity using word embedding.

Capabilities

Given adequate training pairs, this model can learn Semantic as well as structural similarity.
For phrases, the model learns word based embeddings to identify structural/syntactic similarities.

Examples :
	Question1 : What is the step by step guide to invest in share market in india?
	Question2 : What is the step by step guide to invest in share market?
	Result: It is not duplicate
  
	Question1 : Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?
	Question2 : I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about 	me?
	Result: It is  duplicate

Step by step Working:

1. Analyze input dataset with pandas
- Check total number of question pairs for training
- Check duplicate data ratio and percentage
- Combined question1 and question2 in a single list as a series
- Identify unique values in the questions list
- Total number of questions in the training data
- Number of questions that appear multiple times

2. Clean input dataset with natural language processing following techniques:
- Convert all questions in same case (lower or upper)
- Remove of Stop Words from questions
- Remove special characters from questions
- Apply Lemmatization

3. Train Siamese neural network with following steps:
- created questions pairs
- created word embedding meta data
- Fit trainning dataset in the model

4. Test Siamese neural network with following steps:
- Predict output results (is duplicate or not)
- Map output results with input dataset.

Environment:

Keras -> 2.1.6

pandas -> 0.17.1

gensim -> 3.1.0

numpy -> 1.13.1
