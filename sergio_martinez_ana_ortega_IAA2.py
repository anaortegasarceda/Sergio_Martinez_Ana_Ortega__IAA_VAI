import os,shutil
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import pandas as pd
import json

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16, Xception

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from scikeras.wrappers import KerasClassifier

from skimage.io import imread
from skimage.transform import resize
from PIL import ImageMode
from keras.callbacks import Callback


class ANN_Train():

    """Clase realizar la práctica de clasificación de imágenes"""

    def __init__(self, images_configuration , param, cnn_name = 'VGG16', random_seed = 0, experiment_type = 'fit_params'):
        """Contructor donde se fija la random seed y algunos parámetros"""
        tf.random.set_seed(random_seed)
        tf.keras.utils.set_random_seed(random_seed)
        np.random.seed(random_seed)
        self.GPU_configured = True
        self.img_config = images_configuration
        self.param = param
        self.random_seed = 0
        self.cnn_name = cnn_name
        self.cnn = None
        self.experiment_type = experiment_type

    def setup_gpu(self):
        """Método para configurar la GPU, de existir"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=7500)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            self.GPU_configured = True
        else:
            print("GPU not found")

    def setup_dataset(self):
        """Método para configurar el dataset, de forma que sea apto para introducir en los modelos"""
        main_path = self.img_config['main_path']
        image_size = self.img_config['image_size']
        channels = self.img_config['channels']

        classnames = os.listdir(main_path)
        print(classnames)
        for each_class in classnames:
            print(f"Class: {each_class}, has {len(os.listdir(os.path.join(main_path, each_class )))} samples")


        X = []
        y_str = []
        for classname in classnames:
            source_dir = os.path.join(main_path,classname)
            print(f'source_dir: {source_dir}')
            for imagefile in os.listdir(source_dir):
                image = imread(os.path.join(source_dir,imagefile))
                image = resize(image, (image_size,image_size,channels))
                X.append(image)
                y_str.append(classname)

        X = np.array(X)
        y_str = np.array(y_str)

        encoder = preprocessing.LabelEncoder()
        y = encoder.fit_transform(y_str)
        self.classes = encoder.classes_

        return X, y

    def setup_cnns(self):
        """Configuración de las diferentes redes convolucionales a usar"""
        if self.cnn_name == 'VGG16':
          self.cnn = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif self.cnn_name == 'Xception':
          self.cnn = Xception(weights='imagenet', include_top=False, input_shape=(229, 229, 3))
        else:
          print("CNN not found")

        if self.cnn:
          print(self.cnn_name + " selected")
          for layer in self.cnn.layers[:]:
              layer.trainable = False

          for layer in self.cnn.layers:
              print(layer, layer.trainable)

    def plot_history(self, history, parameter):
        """Método para imprimir las gráficas de accuracy y loss en el barrido de parámetros"""
        plt.figure(figsize=(15,5))
        plt.plot(history.history_[parameter], label=parameter)
        plt.plot(history.history_['val_'+parameter], label='Val '+parameter)
        plt.xlabel('Epoch')
        plt.ylabel('Error [MPG]')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/accuracy_loss/' + self.cnn_name + "_" + str(self.param) + "_" + parameter + '.jpg')

    def print_confusion_matrix(self, y_test, y_predict):
        """Método para imprimir las matrices de confusión en el caso del barrido de parámetros"""
        conf_matrix = confusion_matrix(y_test, y_predict)
        disp_knn = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp_knn.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de confusión")
        plt.savefig('results/c_matrix/' + self.cnn_name + "_" + str(self.param) + "_confusion.jpg")


    def create_model(self, pretrained_model, n_layers, n_neurons, activation, dropout_rate):
        """Método para crear un modelo Sequential de Keras"""
        image_size = self.img_config['image_size']
        channels = self.img_config['channels']
        [n_neurons] if type(n_neurons) is int else n_neurons

        model = models.Sequential()

     
        model.add(layers.InputLayer(shape=(image_size, image_size, channels)))
        model.add(layers.Resizing(image_size, image_size))
        model.add(layers.RandomFlip("horizontal_and_vertical"))
        model.add(layers.RandomRotation(0.2))


        model.add(pretrained_model)
        model.add(layers.Flatten())
        for i in range(n_layers):
            model.add(layers.Dense(n_neurons[i], activation=activation))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.summary()
        return model


    def train_and_valuate_params(self, x_train, y_train, x_test, y_test, pretrained_model, layers = 1, neurons = 256, batch_size = 100, activation = 'relu', dropout_rate = 0.0, learning_rate = 0.01, tolerance = 0, epochs = 100, patience = 10):
        """Método para realizar el barrido de parámetros"""
        early_stopping = callbacks.EarlyStopping(monitor='loss', min_delta = tolerance, verbose = 1, restore_best_weights = True, patience = patience)
        model = KerasClassifier(model = self.create_model(pretrained_model=pretrained_model, n_layers=layers, n_neurons=neurons, activation=activation, dropout_rate=dropout_rate),
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=1,
                                validation_split=0.2,
                                optimizer=tf.optimizers.Adam(learning_rate),
                                loss = 'binary_crossentropy',
                                metrics=["accuracy"],
                                callbacks=[early_stopping]
                                )
        ini = time.time()
        history = model.fit(x_train, y_train)
        t_train = round((time.time() - ini)/60, 3)
        print(f"Tiempo entrenamiento = {t_train} min")
        accuracy = model.score(x_test, y_test)

        y_predict = model.predict(x_test)
        self.print_confusion_matrix(y_test, y_predict)
        self.plot_history(history, 'accuracy')
        self.plot_history(history, 'loss')

        return accuracy, t_train

    def fit_params(self, X, y):
        """Método para lanzar el barrido de parámetros"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_seed)

        results = []

        # Probar diferentes valores de cada parámetro mientras se mantienen los demás constantes
        accuracy, t_train = self.train_and_valuate_params(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test, pretrained_model=self.cnn, **self.param)
        results.append((self.param.keys(), self.param.values(), accuracy, t_train))


        # Convertir resultados a DataFrame para análisis
        df_results = pd.DataFrame(results, columns=['parameter', 'value', 'accuracy', 'train_time'])

        print(df_results)

    
    def cross_validation(self, X, y, pretrained_model, layers, neurons, batch_size, activation, dropout_rate, learning_rate, tolerance, epochs, patience):
        """Método para realizar el entrenamiento final, con validación cruzada"""
        early_stopping = callbacks.EarlyStopping(monitor='loss', min_delta = tolerance, verbose = 1, restore_best_weights = True, patience = patience)
        model = KerasClassifier(model = self.create_model(pretrained_model=pretrained_model, n_layers=layers, n_neurons=neurons, activation=activation, dropout_rate=dropout_rate),
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=1,
                              optimizer=tf.optimizers.Adam(learning_rate),
                              loss = 'binary_crossentropy',
                              metrics=["accuracy"],
                              callbacks=[early_stopping]
                              )

        #Entrenamiento con 2 KFold y cross validation
        kf_sh = KFold(n_splits=2, shuffle=True, random_state=self.random_seed)
        ini = time.time()
        res_cv_sh = cross_validate(model,X,y,cv=kf_sh, return_estimator=True)

        print(f"Tiempo entrenamiento = {(time.time() - ini)/60:.3f} min")

        #Generación de informe, histograma y matriz de confusión
        test_sets = []
        for train_index, test_index in kf_sh.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            test_sets.append((X_test,y_test))

        estimators = res_cv_sh['estimator']
        conf_matrix = []
        clf_report = []
        for i in range(len(estimators)):
            estimator = estimators[i]
            y_predict = estimator.predict(test_sets[i][0])
            y_expected = test_sets[i][1]
            conf_matrix.append(confusion_matrix(y_expected, y_predict))
            clf_report.append(classification_report(y_expected, y_predict, output_dict=True, zero_division=1))

        print(clf_report)

        recall = np.zeros(len(self.classes))
        accuracy = []
        for dic in clf_report:
            for i in range(len(self.classes)):
                recall[i] = recall[i] + dic[str(i)]['recall']
            accuracy.append(dic['accuracy'])
        for i in range(len(self.classes)):
            recall[i] = recall[i] / len(clf_report)

        accuracy = np.array(accuracy) 
        mean_acc = accuracy.mean()
        std_acc = accuracy.std()

        accuracy_list = [accuracy]
        label_list = ["(256) - batch:100"]
        plt.boxplot(accuracy_list, labels=label_list)
        plt.ylabel('Accuracy')
        plt.xlabel('Configuración de red')
        plt.ylim(0,1)  


        plt.savefig(f'results/Xception_Hist_layers_{layers}_neurons_{neurons}_activation_{activation}.jpg')
        plt.close()

        print("\n---------------------------------------------")
        print(" Exactitud y sensibilidad media")
        print("---------------------------------------------\n")

        print("Modelo                 ", end='')
        print("Exactitud          ", end='')
        for i in range(len(self.classes)):
            print(f"{self.classes[i]:15}", end='')
        linea = "-"*130
        print(f"\n{linea}")
        print(f"(256) - batch: 100  ", end='')
        print(f"{mean_acc:4.2f}(+/- {std_acc:4.2f})", end='')
        for i in range(len(self.classes)):
            print(f"{recall[i]:15.2}", end='')
        print()
        print()

        avg_cm = np.zeros(conf_matrix[0].shape)
        for matrix in conf_matrix:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    avg_cm[i,j] +=  matrix[i,j]
        for i in range(avg_cm.shape[0]):
            for j in range(avg_cm.shape[1]):
                avg_cm[i,j] = int(round(avg_cm[i][j]/len(conf_matrix)))

        print(avg_cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=avg_cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de confusion")
        plt.savefig(f'results/Xception_conf_layers_{layers}_neurons_{neurons}_activation_{activation}.jpg')
        plt.close()

    def launch_cross_validation(self, X, y):
        """
        Lanzamiento de las combinaciones finales
        Se ha optado por indicarlas de forma manual para evitar complicaciones como con el json del barrido.
        """
        layers = [2,3]
        neurons = {
            '1':[[256], [128], [64]],
            '2':[[128,64], [64,32], [32,16]],
            '3':[[64,32,16], [32,16,8], [16,8,4]]
        }
        activations = ['relu', 'tanh']

        for layer in layers:
            for neuron in neurons[str(layer)]:
                for activation in activations:
                    f = open(f"results/Xception_cv_layers_{layer}_neurons_{neuron}_activation_{activation}_output.txt", "w")
                    sys.stdout = f
                    self.cross_validation(X=X, y=y, pretrained_model=self.cnn, layers=layer, neurons=neuron, batch_size=100, activation=activation, dropout_rate=0.0, learning_rate=0.001, tolerance=0.1, epochs=50, patience=5)
                    f.close()


    def run_experiment(self):
        """Método para lanzar el experimento"""
        self.setup_gpu()
        self.setup_cnns()
        if self.GPU_configured and self.cnn:
            X, y = self.setup_dataset()
            if self.experiment_type == 'fit_params':
                self.fit_params(X, y)
            elif self.experiment_type == 'cross_validation':
                self.launch_cross_validation(X,y)
            else:
                print("Experiment not found")

        else:
            print("No GPU. Finishing...")

def main():
    """
    Función principal y definición de parámetros
    El barrido de parámetros se hacia a partir de un json para tratar de hacer 
    el código más genérico pero no fue muy efectivo
    """
    experiment_type = 'cross_validation'
    cnn_name = 'Xception'
    images_config = {
    'main_path':'train/',
    'image_size':229,
    'channels':3
    }

    sys_original = sys.stdout

    param_file_name = 'params.json'

    param_file = open(param_file_name, 'r')
    params = json.load(param_file)

    if experiment_type == 'fit_params':
      for param in params:
          f = open(f"outputs/{cnn_name}_{str(param)}_output.txt", "w")
          sys.stdout = f
          ann = ANN_Train(images_config, param, cnn_name, experiment_type = experiment_type)
          ann.run_experiment()
          f.close()
    else:
          ann = ANN_Train(images_config, param={}, cnn_name = cnn_name, experiment_type = experiment_type)
          ann.run_experiment()


    sys.stdout = sys_original
    param_file.close()

if __name__ == '__main__':
    main()
