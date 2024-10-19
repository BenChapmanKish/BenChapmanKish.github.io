#!/usr/bin/python3
# ECE 498B 2022
# Ben Chapman-Kish

# Automated selection of optimal model for task based on specified metrics

from __future__ import annotations
from shared.types import *
from engine.models import BaseModel, ModelType, KerasCustomClassifier, KerasCustomRegressor, SVMClassifier, SVMRegressor

from sortedcontainers import SortedList
from timeit import default_timer
from socket import gethostname
from keras.losses import Loss

#from sklearn.model_selection import KFold, train_test_split
#import matplotlib.pyplot as plt

KERAS_RESULTS_FILE="./results/keras_model_train_results.csv"
SVM_RESULTS_FILE="./results/svm_model_train_results.csv"

class ModelSelector(object):
    def __init__(self, job_models: list[BaseModel], course_models: list[BaseModel]):
        self.models = {
            PostingType.JOB: job_models,
            PostingType.COURSE: course_models
        }

        self.current_best_model: dict[PostingType, Optional[BaseModel]] = {
            PostingType.JOB: None,
            PostingType.COURSE: None
        }
    
    def learn(self, record_type: PostingType, x_data: np.ndarray, y_data: np.ndarray) -> None:
        best_model = None
        best_acc = 0

        num_reviews = len(y_data)
        hostname = gethostname().split('.')[0]
        logstr = ""
        random_state = round(default_timer() * 100)
        model_n = 0

        log.info(f"Training {len(self.models[record_type])} {record_type.name} models on {num_reviews} reviews")

        KERAS_RESULTS_CSV_FILENAME = f"./results/keras_results_{num_reviews}_{hostname}_{random_state}.csv"
        SVM_RESULTS_CSV_FILENAME = f"./results/svm_results_{num_reviews}_{hostname}_{random_state}.csv"

        keras_csv = open(KERAS_RESULTS_CSV_FILENAME, 'x')
        keras_csv.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            'model_num',
            'model_type',
            'learning_rate',
            'num_layers',
            'loss_func',
            'num_layer_steps',
            'units_per_layer',
            'dropout_rate',
            'normalize',
            'accuracy',
            'loss',
            'num_epochs',
            'train_time'
        ))

        svm_csv = open(SVM_RESULTS_CSV_FILENAME, 'x')
        svm_csv.write("{},{},{},{},{},{},{}\n".format(
            'model_num',
            'model_type',
            'kernel',
            'C',
            'gamma',
            'accuracy',
            'train_time'
        ))

        for model in self.models[record_type]:
            model_n += 1
            timing_start = default_timer()
            
            result = model.fit(x_data, y_data, random_state=random_state)
            
            timing_end = default_timer()
            elapsed = timing_end - timing_start

            acc = result.accuracy if isinstance(result, TrainingResult) else result
            if acc > best_acc:
                best_acc = acc
                best_model = model
            
            if isinstance(model, (KerasCustomClassifier, KerasCustomRegressor)):
                info = model.info_dict

                if not isinstance(result, TrainingResult):
                    log.error(f"Error: Result of Keras model should be type TrainingResult, not {type(result)}")
                    continue

                model_name = (str(model) if model.name is None else f"{model.name}" + (" " if model.type==ModelType.REGRESSOR else ""))
                logstr = f"[#{model_n:03d}] {model_name} has accuracy: {acc:6.2%} (time: {elapsed:6.3f}s) (# epochs: {result.num_epochs:2d}, loss: {result.loss:.2f})"
                loss_func = model.loss_func.name if isinstance(model.loss_func, Loss) else model.loss_func
        
                keras_csv.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    str(model_n),
                    model.type.name,
                    info.get('learning_rate', str(model.learning_rate)),
                    info.get('num_layers', str(model.num_layers)),
                    info.get('loss_func', str(loss_func)),
                    info.get('num_layer_steps', ''),
                    info.get('units_per_layer', ''),
                    info.get('dropout_rate', ''),
                    info.get('normalize', ''),
                    f"{acc:.5}",
                    f"{result.loss:.4f}",
                    str(result.num_epochs),
                    f"{elapsed:.4f}"
                ))
            
            elif isinstance(model, (SVMClassifier, SVMRegressor)):
                model_name = (str(model) if model.name is None else f"{model.name}" + (" " if model.type==ModelType.REGRESSOR else ""))
                logstr = f"[#{model_n:03d}] {model_name} has accuracy: {acc:6.2%} (time: {elapsed:6.3f}s)"

                svm_csv.write("{},{},{},{},{},{},{}\n".format(
                    str(model_n),
                    model.type.name,
                    model.kernel,
                    str(model.C),
                    str(model.gamma),
                    f"{acc:.5}",
                    f"{elapsed:.4f}"
                ))
            
            log.info(logstr)
        
        keras_csv.close()
        svm_csv.close()
        
        log.info(f"Old best {record_type.name} model: {self.current_best_model[record_type]}")
        log.info(f"New best {record_type.name} model: {best_model} (acc: {best_acc:.2%})")
        log.info("")

        self.current_best_model[record_type] = best_model
    
    # O(p*log(p)) sorting function
    def _sort_results(self, pred_scores: np.ndarray, model_type: ModelType) -> list[tuple[int, float]]:
        results = SortedList(key=lambda x: x[1])

        # O(p) iteration over each posting's score
        if model_type == ModelType.CLASSIFIER:
            for index, scores in enumerate(pred_scores):
                # if np.argmax(scores) < 4: continue
                
                pred = (index, max(scores[-1], scores[-2]/1.5))
                results.add(pred) # O(log(p)) bisect-sorted list insertion

        elif model_type == ModelType.REGRESSOR:
            for index, score in enumerate(pred_scores):
                # if score < 4.0: continue

                pred = (index, score)
                results.add(pred) # O(log(p)) bisect-sorted list insertion
        
        return list(results)
    
    def predict(self, record_type: PostingType, x_data: np.ndarray) -> list[tuple[int, float]]:
        # Returns pre-sorted list in ascending score order
        model = self.current_best_model[record_type]
        assert(model is not None)

        log.info(f"Making predictions with model: {model}")
        log.debug(f"Data to predict: \n{x_data}")

        timing_start = default_timer()

        pred_scores = model.predict(x_data) # O(p) heaviest step (ML prediction)
        
        timing_end = default_timer()
        elapsed = timing_end - timing_start

        log.info(f"[Suggestion Generation Timing] Prediction generation time: {elapsed*1000:.0f}ms")
        log.debug(f"Predicted scores: \n{pred_scores}")

        results: list[tuple[int, float]] = []

        timing_start = default_timer()

        results = self._sort_results(pred_scores, model.type)
        
        timing_end = default_timer()
        elapsed = timing_end - timing_start

        log.info(f"[Suggestion Generation Timing] Prediction sorting time: {elapsed*1000:.0f}ms")
        
        return results


    
