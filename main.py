import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

text = input("Enter text: ")
tokenizer = Tokenizer()
text = text.lower()

tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index)
print("Total no. of words = ",total_words)

tokenizer.word_index
# print(tokenizer.word_index)

input_sequences = []
for line in text.split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# print(input_sequences)
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X = input_sequences[:,:-1]
Y = input_sequences[:,-1]

Y = tf.keras.utils.to_categorical(Y, num_classes=total_words+1)

model = Sequential()
model.add(Embedding(total_words+1, 10, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words+1, activation='sigmoid'))
#model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, epochs=10, verbose=1)

input_text = input("Enter text: ")
next_words = 10
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    input_text += " " + output_word

print(input_text)
