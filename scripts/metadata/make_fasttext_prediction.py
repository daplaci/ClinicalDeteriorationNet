import fasttext
import subprocess
from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_score, recall_score, confusion_matrix
import numpy as np
import os
import json
import sys
import re
import argparse

parser = argparse.ArgumentParser(description='train fasttext model')
parser.add_argument('--data', type=str, default="diag")
parser.add_argument('--rerun', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)

args = parser.parse_args()
data = args.data
label_name = "__label__1.0"
len_label = len(label_name)
train_file = f"input/train_{data}.txt"
pred_file = f"input/pred_{data}.txt"
model_name = f"model_{data}.bin"
model_path = f"input/{model_name}"

if (model_name not in os.listdir('input/')) or args.rerun:
	print (f"Generating model...{data}\n")
	model = fasttext.train_supervised(input=train_file, 
		autotuneValidationFile=f"input/val_{data}.txt", 
		autotuneMetric=f"f1:{label_name}", 
		autotuneDuration=600)

	model.save_model(model_path)
else:
	print ("Warning: using a model already trained. USe --rerun to overwrite with a new model")

if args.test:
	val_file = f"input/test_{data}.txt"
else:
	val_file = f"input/val_{data}.txt"

sys.exit(0)
subprocess.call(f"/fasttext predict-prob {model_path} {val_file} > {pred_file} 2>&1", shell=True)

preds = []
with open(pred_file, 'r') as pf:
	for l in pf:
		label, prob = l.strip("\n").split(" ")
		if label_name in label:
			preds.append(float(prob))
		else:
			preds.append(1-float(prob))
preds = np.array(preds)

target = []
with open(val_file, 'r') as pf:
	for l in pf:
		prob = label_name in l[:len_label]
		target.append(float(prob))
target = np.array(target)

metrics_dict = {}
metrics_dict.update({'auc':roc_auc_score(target, preds)}) 
preds = (preds > 0.1).astype(np.int32)
metrics_dict.update({'cm':str(confusion_matrix(target, preds))}) 
metrics_dict.update({'mcc':matthews_corrcoef(target, preds)})
metrics_dict.update({'precision':precision_score(target, preds, zero_division=0)})
metrics_dict.update({'recall':recall_score(target, preds, zero_division=0)}) 
print (json.dumps(metrics_dict, indent=2))

