

import numpy as np
import pandas as pd


# HYPERPARAMETERS
input_size = 10
output_size = 100
hidden_layers_sizes = 10 # Tried to choose not too large number
learning_rate = 0.01 # Choose not a very small or very large number
number_of_epochs = 1
train_data_path = "./data/drugLibTrain_raw.tsv"  # please use relative path like this
test_data_path = "./data/drugLibTest_raw.tsv"  # please use relative path like this

#Initial Values of weights and biases
W_0 = {
    'W1': np.random.randn(hidden_layers_sizes, 16492) * np.sqrt(2 / hidden_layers_sizes),
    'b1': np.ones((16492, 1)) * 0.01,
    'W2': np.random.randn(10, hidden_layers_sizes) * np.sqrt(2 / hidden_layers_sizes),
    'b2': np.ones((10, 1)) * 0.01
}

#Class for creating one-hot encoding matrix from scratch
class Vocabulary:

    def __init__(self, name):
        PAD_token = 0  # Used for padding short sentences
        SOS_token = 1  # Start-of-sentence token
        EOS_token = 2  # End-of-sentence token
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word): #Adding word
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1

    def add_sentence(self, sentence): #Adding sentence. Splits given sentence with space and uses add_word for each splitted word.
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index): #Gives the word of the given index.
        return self.index2word[index]

    def to_index(self, word): #Gives the index of the given word.
        return self.word2index[word]

    def get_num_sentences(self): #Returns number of sentences
        return self.num_sentences

    def get_num_words(self): #Returns number of words
        return self.num_words


voc = Vocabulary('test') #Creating Vocabulary Class' Object

data_train = pd.read_csv("./data/drugLibTrain_raw.tsv", sep='\t') #Reading the dataset
train_x = [data_train[
               'commentsReview']] #Choosing the column of the dataset

sent_tkns = []
listed_x = train_x[0].tolist()
special_chars = [",", ".", ":", "!", "?", ";", "(", ")", "[", "]"]

for i in listed_x:
    try:
        for word in i.split(' '):
            if word not in special_chars: #Checking if the splitted text have special characters.
                sent_tkns.append(word)
    except:
        continue

for i in sent_tkns:
    voc.add_word(i)


def activation_function(layer):
    # The activation function is Sigmoid function
    return 1 / (1 + np.exp(-layer))


def derivation_of_activation_function(signal):
    # derivative of Sigmoid function
    f = 1 / (1 + np.exp(-signal))
    df = (1 - f).T * f
    return df


def loss_function(true_labels, probabilities):
    x = probabilities
    x = np.array(x)
    true_labels = np.array(true_labels, dtype=int)
    # Sum-of-Squares Error (RSS) is used to turn activations into probability distribution
    result = np.sum(np.square((x[np.arange(1), true_labels.argmax(axis=0)]) - true_labels))
    loss = result
    return loss


def derivation_of_loss_function(true_labels, probabilities):
    # the derivation should be with respect to the output neurons
    x = probabilities
    x = np.array(x)
    true_labels = np.array(true_labels, dtype=int)
    res = 2 * np.sum((x[np.arange(1), true_labels.argmax(axis=0)]) - true_labels)
    return res


def forward_pass(data):
    one_hot_encoding_of_a_sentence = []
    for i in data:
        try:
            for word in i.split(' '):
                x = np.zeros(voc.get_num_words())
                x[voc.to_index(word)] = 1
                one_hot_encoding_of_a_sentence.append(x)
            break
        except:
            continue

    one_hot_encoding_of_a_sentence_matrix = np.matrix(one_hot_encoding_of_a_sentence) #One-hot encoding for the given "data"
    z1 = np.dot(one_hot_encoding_of_a_sentence_matrix, W_0['W1'].T)
    z1 += W_0['b1'].T
    a1 = np.tanh(z1)
    z2 = np.dot(a1, W_0['W2'].T) + W_0['b2'].T
    a2 = activation_function(z2)

    y_pred = activation_function(z2)
    forward_result = {"Z1": z1,
                      "A": a2,
                      "Z2": z2,
                      "Y_pred": y_pred}

    return forward_result


# [hidden_layers] is not an argument, but it is up to you how many hidden layers to implement.
# so replace it with your desired hidden layers 
def backward_pass(input_layer, output_layer, loss):
    one_hot_encoding_of_a_sentence = []
    for i in input_layer:
        try:
            for word in i.split(' '):
                x = np.zeros(voc.get_num_words())
                x[voc.to_index(word)] = 1
                one_hot_encoding_of_a_sentence.append(x)
            break
        except:
            continue

    one_hot_encoding_of_a_sentence_matrix = np.matrix(one_hot_encoding_of_a_sentence)
    output_delta = loss
    z2_delta = np.dot(output_delta, W_0['W2'])
    a_delta = z2_delta * derivation_of_activation_function(output_layer)

    # Updating the Weights and Biases
    W_0["W2"] -= learning_rate * np.dot(output_delta.T, output_layer)
    W_0["b2"] -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)
    W_0["W1"] -= learning_rate * np.dot(a_delta.T, one_hot_encoding_of_a_sentence_matrix)
    W_0["b1"] -= learning_rate * np.sum(a_delta, axis=1)


def train(train_data, train_labels, valid_data, valid_labels):
    for epoch in range(number_of_epochs):
        index = 0

        # Same thing about [hidden_layers] mentioned above is valid here also
        for data, labels in zip(train_data, train_labels):
            z1, a, z2, predictions = forward_pass(data)
            loss_signals = derivation_of_loss_function(labels, forward_pass(data).get(predictions))
            backward_pass(data, forward_pass(data).get(a), loss_signals)
            loss = loss_function(labels, predictions)

            if index % 400 == 0:  # at each 2000th sample, we run validation set to see our model's improvements
                accuracy, loss = test(valid_data, valid_labels)
                print("Epoch= " + str(epoch) + ", Coverage= %" + str(
                    100 * (index / len(train_data))) + ", Accuracy= " + str(accuracy) + ", Loss= " + str(loss))

            index += 1

    # return losses


def test(test_data, test_labels):
    avg_loss = 0
    predictions = []
    labels = []

    for data, label in zip(test_data, test_labels):  # Turns through all data
        _, _, _, prediction = forward_pass(data)
        # print("prediction: ", forward_pass(data).get(prediction))
        predictions.append(forward_pass(data).get(prediction))
        labels.append(label)
        avg_loss += np.sum(loss_function(label, forward_pass(data).get(prediction)))

    # Maximum likelihood is used to determine which label is predicted, highest prob. is the prediction
    # And turn predictions into one-hot encoded

    one_hot_predictions = np.zeros(shape=(len(predictions), output_size))
    for i in range(len(predictions)):
        one_hot_predictions[i][np.argmax(predictions[i])] = 1

    predictions = one_hot_predictions

    accuracy_score = accuracy(labels, predictions)

    return accuracy_score, avg_loss / len(test_data)


def accuracy(true_labels, predictions):
    true_pred = 0
    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == np.argmax(true_labels[i]):  # if 1 is in same index with ground truth
            true_pred += 1
    return true_pred / len(predictions)


if __name__ == "__main__":

    train_data = pd.read_csv(train_data_path, sep='\t')
    test_data = pd.read_csv(test_data_path, sep='\t')
    train_x = train_data[
                   'commentsReview']  # use train_data['commentsReview'] or concatenate benefitsReview,
    # sideEffectsReview, and commentsReview
    train_y = train_data['rating']
    test_x = test_data[
                  'commentsReview']  # use test_data['commentsReview'] or concatenate benefitsReview,
    # sideEffectsReview, and commentsReview
    test_y = test_data['rating']

    # creating one-hot vector notation of labels. (Labels are given numeric in the dataset)
    new_train_y = np.zeros(shape=(len(train_y), output_size))
    new_test_y = np.zeros(shape=(len(test_y), output_size))

    for i in range(len(train_y)):
        new_train_y[i][train_y[i]] = 1

    for i in range(len(test_y)):
        new_test_y[i][test_y[i]] = 1

    train_y = new_train_y
    test_y = new_test_y

    # Training and validation split. (%75-%25)
    valid_x = np.asarray(train_x[int(0.75 * len(train_x)):-1])
    valid_y = np.asarray(train_y[int(0.75 * len(train_y)):-1])
    train_x = np.asarray(train_x[0:int(0.75 * len(train_x))])
    train_y = np.asarray(train_y[0:int(0.75 * len(train_y))])

    train(train_x, train_y, valid_x, valid_y)
    print("Test Scores:")
    print(test(test_x, test_y))
