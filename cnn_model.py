from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import numpy as np
import pickle
import os

def build_cnn_model(input_shape, num_classes, filters=32, dense_units=256):
    """Build a CNN model with specified architecture."""
    model = Sequential()
    
    model.add(Conv2D(filters, (3, 3), input_shape=input_shape, activation='relu', name='conv2d_1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'))
    model.add(Conv2D(filters, (3, 3), activation='relu', name='conv2d_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(units=dense_units, activation='relu', name='dense_1'))
    model.add(Dense(units=num_classes, activation='softmax', name='dense_2'))
    
    return model

def get_optimizer(optimizer_name, learning_rate):
    """Get optimizer with specified learning rate."""
    if optimizer_name == 'adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    else:
        return Adadelta(learning_rate=learning_rate)

def lr_schedule(epoch):
    """Learning rate schedule."""
    return 1.0 if epoch <= 30 else (0.5 if epoch <= 60 else 0.1)

def train_model(model, X_train, y_train, X_test, y_test, optimizer_name, learning_rate, epochs, 
                weights_path, history_path, text_widget=None, main_window=None):
    """
    Train the model with specified parameters.
    Note: weights_path must end with .keras for Keras 3.x compatibility.
    """
    if text_widget:
        text_widget.delete('1.0', 'end')
        text_widget.insert('end', f"Training {optimizer_name} model...\n")
        if main_window:
            main_window.update()
    
    optimizer = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        if text_widget:
            text_widget.insert('end', f"Loaded existing weights for {optimizer_name}\n")
    else:
        model_check_point = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
        hist = model.fit(
            X_train, y_train,
            epochs=epochs,
            shuffle=True,
            verbose=2,
            validation_data=(X_test, y_test),
            callbacks=[model_check_point, LearningRateScheduler(lr_schedule)]
        )
        
        with open(history_path, 'wb') as f:
            pickle.dump(hist.history, f)
        
        return hist.history
    
    return None 