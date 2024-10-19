#!/usr/bin/python3
# ECE 498B 2022
# Ben Chapman-Kish

# Implementations of the trainable models
# NOTE: This file contains only MLP and SVM classifiers and regressors, but that doesn't mean these are the only
# types of models that are appropriate for this task. I would start by trying out a KNN or decisition tree classifier,
# but there could be lots of potential in the use of random forests or gradient-boosted classifiers as well.

from __future__ import annotations
from shared.types import *

import tensorflow.keras as keras # type: ignore
#import keras.layers, keras.models, keras.callbacks, keras.activations, keras.optimizers, keras.initializers, keras.backend
#import keras
#import keras.backend

import sklearn.svm
from sklearn.model_selection import KFold #, train_test_split
from sklearn.metrics import mean_absolute_error
#from sklearn.preprocessing import MinMaxScaler

def create_approx_accuracy(scale: float = 0.5):
    def accuracy(y_true, y_pred):

        y_true_class = keras.backend.argmax(y_true, axis=-1)
        y_pred_class = keras.backend.argmax(y_pred, axis=-1)

        difference = keras.backend.abs(y_true_class - y_pred_class)
        clipped_difference = 2 - keras.backend.clip(difference, 0, 2)
        scaled_difference = keras.backend.cast(clipped_difference, 'float64') * scale
        unit_difference = keras.backend.clip(scaled_difference, 0, 1) # value between 0 and 1 for how good each prediction is

        correct_sum = keras.backend.cast(keras.backend.sum(unit_difference), 'int32')
        total_sum = keras.backend.cast(keras.backend.sum(keras.backend.clip(y_true_class, 1, 1)), 'int32')

        return correct_sum / total_sum
    
    return accuracy

class ModelType(Enum):
    NONE = 0
    CLASSIFIER = 1
    REGRESSOR = 2

class BaseModel(object):
    def __init__(self, type: ModelType, name: Optional[str] = None):
        self.type = type
        self.name = name

    def fit(self, x_data: np.ndarray, y_data: np.ndarray, random_state: Optional[int] = None) -> float | TrainingResult:
        raise NotImplementedError
    
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class KerasBaseModel(BaseModel):
    def __init__(self,
        type: ModelType,
        name: Optional[str] = None,
        
        loss_func: str | keras.losses.Loss = "categorical_crossentropy",
        learning_rate: float = 0.1,
        batch_size: int = 32,
        num_epochs: int = 100,
        shuffle: bool = True,
        validation_ratio: float = 0.1
    ):
        super().__init__(type, name)

        self.loss_func = loss_func
        self.learning_rate = learning_rate

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_ratio = validation_ratio

        self.model: keras.Model = keras.models.Sequential(name=self.name)
        self.callbacks = []

    def fit(self, x_data: np.ndarray, y_data: np.ndarray, random_state: Optional[int] = None) -> TrainingResult:
        if self.type == ModelType.CLASSIFIER:
            from tensorflow.keras.utils import to_categorical # type: ignore
            y_data = to_categorical(y_data - 1, num_classes=5)
        
        losses: list[float] = []
        accs: list[float] = []
        kfold_iteration = 1
        num_epochs = 0

        for train_index, validate_index in KFold(n_splits=Config.validation_split_kfolds, shuffle=self.shuffle, random_state=random_state).split(x_data):
            train_x_data, validate_x_data = x_data[train_index], x_data[validate_index]
            train_y_data, validate_y_data = y_data[train_index], y_data[validate_index]

            if Config.verbose_training: log.info('\n' + '-'*50 + '\n')

            h: keras.callbacks.History = self.model.fit(
                train_x_data,
                train_y_data,
                batch_size=self.batch_size,
                validation_data=(validate_x_data, validate_y_data),
                epochs=self.num_epochs,
                callbacks=self.callbacks,
                shuffle=self.shuffle,
                verbose=int(Config.verbose_training)
            )

            if Config.verbose_training: log.info('\n' + '-'*50 + '\n')

            num_epochs = len(h.history['val_loss'])
            val_loss = h.history['val_loss'][-1]
            
            if self.type == ModelType.CLASSIFIER:
                val_accuracy = h.history['val_accuracy'][-1]
                
            else:
                error = h.history['val_root_mean_squared_error'][-1]
                # Since the loss is the mean squared error between our predicted and actual rating, we can get an
                # accuracy metric by dividing the root mse by the full domain of our output (5-1=4)
                val_accuracy = 1 - (error/4.0)

            log.debug(f"Model {self}: KFold {kfold_iteration} loss: {val_loss:.2f} accuracy: {val_accuracy:.2%} (epochs: {num_epochs})")
            
            losses.append(val_loss)
            accs.append(val_accuracy)
            
            kfold_iteration += 1
        
        avg_loss = sum(losses)/len(losses)
        avg_acc = sum(accs)/len(accs)

        return TrainingResult(num_epochs=num_epochs, loss=avg_loss, accuracy=avg_acc)
    
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        return self.model.predict(x_data)


class KerasDenseClassifier(KerasBaseModel):
    def __init__(self,
        name: Optional[str] = None,

        layers_and_nodes: list[int] = [20],
        activation_func: str = "sigmoid",
        
        loss_func: str = "categorical_crossentropy",
        learning_rate: float = 0.1,
        weight_init_stddev: float = 10.0,
        bias_init: str = "zeros",

        batch_size: int = 32,
        num_epochs: int = 100,
        early_stop_patience: int = 20,
        shuffle: bool = True,
        validation_ratio: float = 0.1
    ):
        super().__init__(ModelType.CLASSIFIER, name, loss_func, learning_rate, batch_size, num_epochs, shuffle, validation_ratio)

        self.layers_and_nodes = layers_and_nodes
        self.activation_func = activation_func

        self.weight_init_stddev = weight_init_stddev
        self.bias_init = bias_init

        #self.model.add(keras.layers.Input(shape=(11,)))

        for num_nodes in layers_and_nodes:
            self.model.add(keras.layers.Dense(
                units=num_nodes,
                activation=activation_func,
                kernel_initializer=keras.initializers.RandomNormal(stddev=weight_init_stddev),
                bias_initializer=bias_init
            ))

        self.model.add(keras.layers.Dense(units=5, activation="softmax"))

        if early_stop_patience:
            self.callbacks = [keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stop_patience,
                restore_best_weights=True
            )]

        self.model.compile(
            loss=loss_func,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
    
    def __repr__(self) -> str:
        namestr = "" if self.name is None else f'"{self.name}", '
        return "KerasDenseClassifier({}layers_and_nodes={}, learning_rate={}, weight_init_stddev={}, bias_init={})" \
            .format(namestr, self.layers_and_nodes, self.learning_rate, self.weight_init_stddev, self.bias_init)

class KerasDenseRegressor(KerasBaseModel):
    def __init__(self,
        name: Optional[str] = None,

        layers_and_nodes: list[int] = [20],
        activation_func: str = "sigmoid",

        loss_func: str = "mean_squared_error",
        learning_rate: float = 0.1,
        weight_init_stddev: float = 10.0,
        bias_init: str = "zeros",

        batch_size: int = 32,
        num_epochs: int = 100,
        early_stop_patience: int = 20,
        shuffle: bool = True,
        validation_ratio: float = 0.1
    ):
        super().__init__(ModelType.REGRESSOR, name, loss_func, learning_rate, batch_size, num_epochs, shuffle, validation_ratio)

        self.layers_and_nodes = layers_and_nodes
        self.activation_func = activation_func

        self.weight_init_stddev = weight_init_stddev
        self.bias_init = bias_init

        #self.model.add(keras.layers.Input(shape=(11,)))

        for num_nodes in layers_and_nodes:
            self.model.add(keras.layers.Dense(
                units=num_nodes,
                activation=activation_func,
                kernel_initializer=keras.initializers.RandomNormal(stddev=weight_init_stddev),
                bias_initializer=bias_init
            ))

        self.model.add(keras.layers.Dense(units=1, activation="linear"))

        if early_stop_patience:
            self.callbacks = [keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stop_patience,
                restore_best_weights=True
            )]

        self.model.compile(
            loss=loss_func,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )

    def __repr__(self) -> str:
        namestr = "" if self.name is None else f'"{self.name}", '
        return "KerasDenseRegressor({}layers_and_nodes={}, learning_rate={}, weight_init_stddev={}, bias_init={})" \
            .format(namestr, self.layers_and_nodes, self.learning_rate, self.weight_init_stddev, self.bias_init)

class KerasCustomClassifier(KerasBaseModel):
    def __init__(self,
        name: Optional[str] = None,
        info_dict: dict[str, str] = {},
        layers: list[keras.layers.Layer] = [keras.layers.Dense(20)],
        
        loss_func: str | keras.losses.Loss = keras.losses.CategoricalCrossentropy(label_smoothing=Config.categorical_label_smoothing),
        learning_rate: float = 0.1,
        metrics: list[str| keras.metrics.Metric] = ['accuracy'],

        batch_size: int = 32,
        num_epochs: int = 100,
        early_stop_patience: int = 20,
        shuffle: bool = True,
        validation_ratio: float = 0.1
    ):
        super().__init__(ModelType.CLASSIFIER, name, loss_func, learning_rate, batch_size, num_epochs, shuffle, validation_ratio)

        #self.model.add(keras.layers.Input(shape=(11,)))

        self.info_dict = info_dict

        self.num_layers = len(layers)
        for layer in layers:
            self.model.add(layer)

        self.model.add(keras.layers.Dense(units=5, activation="softmax"))

        if early_stop_patience:
            self.callbacks = [keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stop_patience,
                restore_best_weights=True
            )]

        self.model.compile(
            loss=loss_func,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=metrics
        )
    
    def __repr__(self) -> str:
        namestr = "" if self.name is None else f'"{self.name}", '
        return "KerasCustomClassifier({}learning_rate={}, num_layers={})" \
            .format(namestr, self.learning_rate, self.num_layers)

class KerasCustomRegressor(KerasBaseModel):
    def __init__(self,
        name: Optional[str] = None,
        info_dict: dict[str, str] = {},
        layers: list[keras.layers.Layer] = [keras.layers.Dense(20)],

        loss_func: str | keras.losses.Loss = "mean_squared_error",
        learning_rate: float = 0.1,
        metrics: list[str| keras.metrics.Metric] = [keras.metrics.RootMeanSquaredError()],

        batch_size: int = 32,
        num_epochs: int = 100,
        early_stop_patience: int = 20,
        shuffle: bool = True,
        validation_ratio: float = 0.1
    ):
        super().__init__(ModelType.REGRESSOR, name, loss_func, learning_rate, batch_size, num_epochs, shuffle, validation_ratio)

        #self.model.add(keras.layers.Input(shape=(11,)))

        self.info_dict = info_dict

        self.num_layers = len(layers)
        for layer in layers:
            self.model.add(layer)

        self.model.add(keras.layers.Dense(units=1, activation="linear"))

        if early_stop_patience:
            self.callbacks = [keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stop_patience,
                restore_best_weights=True
            )]

        self.model.compile(
            loss=loss_func,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=metrics
        )
    
    def __repr__(self) -> str:
        namestr = "" if self.name is None else f'"{self.name}", '
        return "KerasCustomRegressor({}learning_rate={}, num_layers={})" \
            .format(namestr, self.learning_rate, self.num_layers)



class SVMBaseModel(BaseModel):
    def __init__(self, type: ModelType, name: Optional[str] = None, kernel: str = 'rbf', C: int = 50, gamma: float = 10.0, shuffle: bool = True):
        super().__init__(type, name)

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.shuffle = shuffle
        
        self.svm = sklearn.svm.SVC(kernel=kernel, C=C, gamma=gamma, probability=True, verbose=Config.verbose_training)

    def score(self, val_x: np.ndarray, val_y_true: np.ndarray) -> float:
        raise NotImplementedError
    
    def fit(self, x_data: np.ndarray, y_data: np.ndarray, random_state: Optional[int] = None) -> float:
        scores: list[float] = []
        kfold_iteration = 1

        for train_index, validate_index in KFold(n_splits=Config.validation_split_kfolds, shuffle=self.shuffle, random_state=random_state).split(x_data):
            train_x_data, validate_x_data = x_data[train_index], x_data[validate_index]
            train_y_data, validate_y_data = y_data[train_index], y_data[validate_index]

            self.svm.fit(train_x_data, train_y_data)
            score = self.score(validate_x_data, validate_y_data)

            log.debug(f"Model {self}: KFold {kfold_iteration} accuracy: {score:.2%}")
            
            scores.append(score)
            kfold_iteration += 1

        return sum(scores)/len(scores)
    
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class SVMClassifier(SVMBaseModel):
    def __init__(self, name: Optional[str] = None, kernel: str = 'rbf', C: int = 50, gamma: float = 10.0, shuffle: bool = True, approx_accuracy_scale: Optional[float] = None):
        super().__init__(ModelType.CLASSIFIER, name, kernel, C, gamma, shuffle)

        self.approx_accuracy_scale = approx_accuracy_scale
    
    def score(self, val_x: np.ndarray, val_y_true: np.ndarray) -> float:
        if self.approx_accuracy_scale is not None:
            from tensorflow.keras.utils import to_categorical # type: ignore

            val_y_pred = self.svm.predict_proba(val_x)
            val_y_true = to_categorical(val_y_true - 1, num_classes=5)

            assert(val_y_true.shape == val_y_pred.shape)

            y_true_class = np.argmax(val_y_true, axis=-1)
            y_pred_class = np.argmax(val_y_pred, axis=-1)

            difference = np.abs(y_true_class - y_pred_class)
            clipped_difference = 2 - np.clip(difference, 0, 2)
            scaled_difference = clipped_difference * self.approx_accuracy_scale
            unit_difference = np.clip(scaled_difference, 0, 1)

            correct_sum = np.sum(unit_difference)
            total_sum = len(val_y_true)

            return correct_sum/total_sum
        
        else:
            return self.svm.score(val_x, val_y_true)

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        return self.svm.predict_proba(x_data)
    
    def __repr__(self) -> str:
        namestr = "" if self.name is None else f'"{self.name}", '
        return f"SVMClassifier({namestr}kernel={self.kernel}, C={self.C}, gamma={self.gamma})"


class SVMRegressor(SVMBaseModel):
    def __init__(self, name: Optional[str] = None, kernel: str = 'rbf', C: int = 50, gamma: float = 10.0, shuffle: bool = True, convert_loss_to_accuracy: bool = False):
        super().__init__(ModelType.REGRESSOR, name, kernel, C, gamma, shuffle)

        self.convert_loss_to_accuracy = convert_loss_to_accuracy
    
    def score(self, val_x: np.ndarray, val_y_true: np.ndarray) -> float:
        if self.convert_loss_to_accuracy:
            val_y_pred = self.svm.predict(val_x)
            error = mean_absolute_error(val_y_true, val_y_pred)
            acc = 1.0 - (error/4.0)
            return acc

        else:
            return self.svm.score(val_x, val_y_true)

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        return self.svm.predict(x_data)

    def __repr__(self) -> str:
        namestr = "" if self.name is None else f'"{self.name}", '
        return f"SVMRegressor({namestr}kernel={self.kernel}, C={self.C}, gamma={self.gamma})"



# TODO: This is a highly inefficient grid search that doesn't even use sklearn's built-in grid search tools.
# Replace this system with my custom-built stochastic parameter search with genetic programming.

JOB_MODELS: list[BaseModel] = []

if True:
    SHORTLIST_JOB_MODELS: list[BaseModel] = []

    SHORTLIST_JOB_MODELS.append(KerasCustomClassifier(
        name="Classifier_A",
        learning_rate=0.01,
        metrics=[create_approx_accuracy(scale=0.75)],
        
        layers=[
            keras.layers.Dense(units=10, activation='sigmoid'),
            keras.layers.Dense(units=10, activation='sigmoid'),
        ]
    ))

    SHORTLIST_JOB_MODELS.append(KerasCustomRegressor(
        name="Regressor_A",
        learning_rate=0.01,
        
        layers=[
            keras.layers.Dense(units=10, activation='sigmoid'),
            keras.layers.Dense(units=10, activation='sigmoid'),
        ]
    ))

    SHORTLIST_JOB_MODELS.append(KerasCustomClassifier(
        name="Classifier_B",
        learning_rate=0.01,
        metrics=[create_approx_accuracy(scale=0.75)],
        
        layers=[
            keras.layers.BatchNormalization(input_shape=(26,)),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dense(units=40, activation='sigmoid'),
        ]
    ))

    SHORTLIST_JOB_MODELS.append(KerasCustomRegressor(
        name="Regressor_B",
        learning_rate=0.01,
        
        layers=[
            keras.layers.BatchNormalization(input_shape=(26,)),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dense(units=40, activation='sigmoid'),
        ]
    ))

    SHORTLIST_JOB_MODELS.append(KerasCustomClassifier(
        name="Classifier_C",
        learning_rate=0.01,
        metrics=[create_approx_accuracy(scale=0.75)],
        
        layers=[
            keras.layers.BatchNormalization(input_shape=(26,)),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dropout(rate=0.1),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dropout(rate=0.1),
            keras.layers.Dense(units=40, activation='sigmoid'),
        ]
    ))

    SHORTLIST_JOB_MODELS.append(KerasCustomRegressor(
        name="Regressor_C",
        learning_rate=0.01,
        
        layers=[
            keras.layers.BatchNormalization(input_shape=(26,)),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dropout(rate=0.1),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dropout(rate=0.1),
            keras.layers.Dense(units=40, activation='sigmoid'),
        ]
    ))

    SHORTLIST_JOB_MODELS.append(KerasCustomClassifier(
        name="Classifier_D",
        learning_rate=0.01,
        metrics=[create_approx_accuracy(scale=0.75)],
        
        layers=[
            keras.layers.BatchNormalization(input_shape=(26,)),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dense(units=40, activation='sigmoid'),
        ]
    ))

    SHORTLIST_JOB_MODELS.append(KerasCustomRegressor(
        name="Regressor_D",
        learning_rate=0.01,
        
        layers=[
            keras.layers.BatchNormalization(input_shape=(26,)),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(units=40, activation='sigmoid'),
            keras.layers.Dense(units=40, activation='sigmoid'),
        ]
    ))

    SHORTLIST_JOB_MODELS.append(SVMClassifier(
        name="Classifier_E",
        kernel='rbf',
        C=10,
        gamma=10,
        approx_accuracy_scale=0.75
    ))

    SHORTLIST_JOB_MODELS.append(SVMRegressor(
        name="Regressor_E",
        kernel='rbf',
        C=10,
        gamma=10,
        convert_loss_to_accuracy=True
    ))

    SHORTLIST_JOB_MODELS.append(SVMClassifier(
        name="Classifier_F",
        kernel='rbf',
        C=50,
        gamma=100,
        approx_accuracy_scale=0.75
    ))

    SHORTLIST_JOB_MODELS.append(SVMRegressor(
        name="Regressor_F",
        kernel='rbf',
        C=50,
        gamma=100,
        convert_loss_to_accuracy=True
    ))

    JOB_MODELS.extend(SHORTLIST_JOB_MODELS)



EXTENDED_JOB_MODELS: list[BaseModel] = []

for learning_rate in (0.005, 0.01, 0.05, 0.1):
    for num_layer_steps in (2, 3, 4):
        for units_per_layer in (10, 20, 30, 40, 50):
            for dropout_rate in (0, 0.1, 0.2):
                for normalize in (False, True):
                    layers=[]

                    if normalize:
                        layers.append(keras.layers.BatchNormalization(input_shape=(26,)))

                    for i in range(num_layer_steps):
                        layers.append(keras.layers.Dense(units=units_per_layer, activation="sigmoid"))

                        if dropout_rate > 0 and i+1 != num_layer_steps:
                            layers.append(keras.layers.Dropout(rate=dropout_rate))
                    
                    class_info={
                        'learning_rate': str(learning_rate),
                        'num_layer_steps': str(num_layer_steps),
                        'units_per_layer': str(units_per_layer),
                        'dropout_rate': str(dropout_rate),
                        'normalize': str(normalize)
                    }

                    EXTENDED_JOB_MODELS.append(KerasCustomClassifier(
                        info_dict=class_info,
                        learning_rate=learning_rate,
                        metrics=[create_approx_accuracy(scale=0.75)],
                        layers=layers
                    ))

                    reg_info={
                        'learning_rate': str(learning_rate),
                        'num_layer_steps': str(num_layer_steps),
                        'units_per_layer': str(units_per_layer),
                        'dropout_rate': str(dropout_rate),
                        'normalize': str(normalize)
                    }

                    EXTENDED_JOB_MODELS.append(KerasCustomRegressor(
                        info_dict=reg_info,
                        learning_rate=learning_rate,
                        layers=layers
                    ))

JOB_MODELS.extend(EXTENDED_JOB_MODELS)



SVM_JOB_MODELS: list[BaseModel] = []

for kernel in ('linear', 'poly', 'rbf'):
    for C in (1, 10, 20, 30, 50, 100):
        for gamma in (1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3):
            SVM_JOB_MODELS.append(SVMClassifier(
                kernel=kernel,
                C=C,
                gamma=gamma,
                approx_accuracy_scale=0.75
            ))

            SVM_JOB_MODELS.append(SVMRegressor(
                kernel=kernel,
                C=C,
                gamma=gamma,
                convert_loss_to_accuracy=True
            ))

JOB_MODELS.extend(SVM_JOB_MODELS)



COURSE_MODELS: list[BaseModel] = [
    KerasDenseClassifier(),
    KerasDenseRegressor(),
    SVMClassifier(),
    SVMRegressor()
]
