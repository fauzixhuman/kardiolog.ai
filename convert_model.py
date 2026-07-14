import joblib
import json
import os
import re

# 1. Convert the existing pkl model to json format locally
pkl_path = r"d:\Kuliah\projek\kardiolog.ai\models\xgboost_model.pkl"
json_path = r"d:\Kuliah\projek\kardiolog.ai\models\xgboost_model.json"

if os.path.exists(pkl_path):
    print(f"Loading {pkl_path}...")
    try:
        model = joblib.load(pkl_path)
        # XGBClassifier has get_booster()
        if hasattr(model, 'get_booster'):
            model.get_booster().save_model(json_path)
        else:
            model.save_model(json_path)
        print(f"Successfully saved to {json_path}")
    except Exception as e:
        print(f"Error converting model: {e}")

# 2. Update the notebook file
notebook_path = r"d:\Kuliah\projek\kardiolog.ai\notebooks\Train_Ensemble_CRNN_XGBoost_Colab.ipynb"
if os.path.exists(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                original_line = line
                # Replace joblib dump for xgboost
                if 'joblib.dump' in line and ('xgb' in line or 'xgboost' in line) and '.pkl' in line:
                    line = re.sub(r"joblib\.dump\((.*?),\s*['\"](.*?)xgboost_model\.pkl['\"]\)", r"\1.save_model('\2xgboost_model.json')", line)
                if line != original_line:
                    modified = True
                new_source.append(line)
            cell['source'] = new_source

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook successfully updated!")
    else:
        print("Could not find the target string in the notebook.")
