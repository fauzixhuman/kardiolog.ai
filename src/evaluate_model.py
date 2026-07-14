"""
evaluate_model.py
-----------------
Fungsi-fungsi untuk mengevaluasi performa klasifikasi 
menggunakan metrik MAcc juri PhysioNet.
"""

def calculate_macc(y_true, y_pred):
    """Menghitung metrik juri (Modified Accuracy)"""
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    macc = (sensitivity + specificity) / 2
    return macc, sensitivity, specificity
