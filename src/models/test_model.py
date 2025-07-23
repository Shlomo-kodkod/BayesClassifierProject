import pandas as pd
from src.models.naive_bayes import NaiveBayes
from src.models.predictor import Predictor


# Class for testing the model accuracy on new data
class TestModel:
    def __init__(self, data: pd.DataFrame, model: NaiveBayes):
        self.__test_data = data
        self.__model = model
        self.__predictor = Predictor(model)

    # Test the model accuracy on the test data
    def test_model(self) -> float:
        correct_cnt = 0
        for index, row in self.__test_data.iterrows():
            current_row = row.drop(self.__model.target_column)
            correct_value = row[self.__model.target_column]
            correct_cnt += (1 if self.__predictor.predict(current_row) == correct_value else 0)
        return (correct_cnt / len(self.__test_data)) * 100
    
    # Calculate the confusion matrix for the model predictions
    def confusion_matrix(self) -> dict | str:
        TP, TN, FP, FN = 0, 0, 0, 0
        option = list(self.__model.target_value.keys())
        if len(option) != 2:
            return "Confusion matrix can only be calculated for binary classification problems."
            
        for index, row in self.__test_data.iterrows():
            current_row = row.drop(self.__model.target_column)
            actual = row[self.__model.target_column]
            predicted = self.__predictor.predict(current_row)
            if actual == option[0] and predicted == actual: TP += 1
            elif actual == option[0] and predicted != actual: FN += 1
            elif actual == option[1] and predicted == actual: TN += 1
            elif actual == option[1] and predicted != actual: FP += 1
        
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

        return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}

    


         
        






