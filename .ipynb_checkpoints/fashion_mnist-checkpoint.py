import tensorflow as tf
import tensorflow.keras as Keras
import numpy as np
import json
import pandas as pd
import argparse
import os


def model(x_train, y_train, x_test, y_test, epochs=1):
    """ Generate a model """
    model = Keras.Sequential()
    model.add(Keras.layers.Flatten(input_shape=[28,28]))
    model.add(Keras.layers.Dense(300, activation='relu'))
    model.add(Keras.layers.Dense(100, activation='relu'))
    model.add(Keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss=Keras.lossses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs)
    model.evaluate(x_test, y_test)
    return model

def _load_datat(base_dir):
    """ Load MNIST Training and testing data """
    data = np.load(os.path.join(base_dir, 'fashion_mnist.npy'), allow_pickle=True)
    return data

def _parse_args():
    parser = argparse.ArgumentParser()
    
     # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--sm-model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
     # Training Parameters, given
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    return parser.parse_known_args()
 
#List devices available to TensorFlow
from tensorflow.python.client import device_lib

if __name__ == "__main__":
    args, unknow = _parse_args()
    print(args)
    print('\n\nDEVICES\n\n')
    print(device_lib.list_local_devices())
    print('\n\n')
    
    print('Loading data..\n')
    (train_data, train_labels), (eval_data, eval_labels) = _load_data(args.train)
    print('Training model for {} epochs..\n\n'.format(args.epochs))
    
    classifier = model( train_data, train_labels, eval_data, eval_labels, epochs=args.epochs)
    
    if args.current_host == args.hosts[0]:
        # Save model in SavedModel format
        classifier.save(os.path.join(args.sm_model_dir, 'my_model.h5'))
        
        # Save model in Keras h5 format
        classifier.save(os.path.join(args.sm_model_dir, 'classifier'))