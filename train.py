import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing import sequence

def train_and_save_model():
    # Parameters
    max_features = 10000
    maxlen = 500
    batch_size = 32
    embedding_dims = 32
    hidden_units = 32

    # Load IMDB dataset
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    
    print('Pad sequences...')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    # Create model
    print('Building model...')
    model = Sequential([
        Embedding(max_features, embedding_dims, input_length=maxlen),
        SimpleRNN(hidden_units),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    print('Training model...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=3,  # Reduced epochs for faster deployment
              validation_data=(x_test, y_test))

    # Save model
    print('Saving model...')
    model.save('simpleRNN.h5')
    return model

if __name__ == "__main__":
    train_and_save_model()
