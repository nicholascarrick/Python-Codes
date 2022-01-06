from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

def evaluate_model(y_true,y_pred):
  metrics_dict = {}

  accuracy = round((accuracy_score(y_true,y_pred))*100,2)
  metrics_dict["accuracy"] = str(accuracy) + "%"

  precision = round((precision_score(y_true,y_pred,average="micro"))*100,2)
  metrics_dict["precision"] = str(precision) + "%"

  recall = round((recall_score(y_true,y_pred,average="micro"))*100,2)
  metrics_dict["recall"] = str(recall) + "%"

  f1 = round((f1_score(y_true,y_pred,average="micro"))*100,2)
  metrics_dict["f1_score"] = str(f1) + "%"

  confuse = confusion_matrix(y_true,y_pred)

  return metrics_dict , confuse
