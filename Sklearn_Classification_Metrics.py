from sklearn.metrics import *

def evaluate_preds(model_name,y_pred,y_true):
  metrics_dict = {}

  accuracy = accuracy_score(y_true,y_pred)
  metrics_dict["accuracy"] = accuracy

  precision = precision_score(y_true,y_pred)
  metrics_dict["precision"] = precision

  recall = recall_score(y_true,y_pred)
  metrics_dict["recall"] = recall

  f1 = f1_score(y_true,y_pred)
  metrics_dict["f1_score"] = f1

  return metrics_dict
