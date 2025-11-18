import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, balanced_accuracy_score
)
from math import sqrt

# ---------- CONFIG ----------
zip_path = r"Prediction.zip"  # Path to zip of predictions
ground_truth_path = r"truth.tsv"  # Ground truth TSV
output_xlsx = r"summary_metrics.xlsx"  # Output Excel file
confusion_folder = r"confusion_matrices"  # Folder for confusion matrices

os.makedirs(confusion_folder, exist_ok=True)

extract_path = os.path.join(os.path.dirname(zip_path), "predictions_extracted")
os.makedirs(extract_path, exist_ok=True)

# ---------- EXTRACT ----------
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

# ---------- LOAD GROUND TRUTH ----------
gt = pd.read_csv(ground_truth_path, sep="\t")
gt_dict = dict(zip(gt["id"], gt[["hate_type", "hate_severity", "to_whom"]].values))

# ---------- HELPER FUNCTION ----------
def calculate_metrics(y_true, y_pred, labels):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)  # Use standard accuracy to match scorer
    bal_acc = balanced_accuracy_score(y_true, y_pred)  # Keep balanced accuracy for BER
    ber = 1 - bal_acc

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Class-wise metrics
    cls_prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    cls_rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    cls_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    
    # Calculate TPR, FPR, Specificity, G-Score per class
    tpr = []
    fpr = []
    specificity = []
    g_score = []
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = sum(cm[i, :]) - tp
        fp = sum(cm[:, i]) - tp
        tn = cm.sum() - (tp + fn + fp)
        
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        spec_val = tn / (tn + fp) if (tn + fp) > 0 else 0
        g_val = sqrt(tpr_val * spec_val)
        
        tpr.append(tpr_val)
        fpr.append(fpr_val)
        specificity.append(spec_val)
        g_score.append(g_val)

    metrics = {
        "Precision": precision,
        "Recall": recall,
        "Micro F1-Score": micro_f1,
        "Macro F1-Score": macro_f1,
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "Balanced Error Rate": ber,
    }
    
    for i, label in enumerate(labels):
        metrics[f"Class-wise Precision ({label})"] = cls_prec[i]
        metrics[f"Class-wise Recall ({label})"] = cls_rec[i]
        metrics[f"Class-wise F1 ({label})"] = cls_f1[i]
        metrics[f"TPR ({label})"] = tpr[i]
        metrics[f"FPR ({label})"] = fpr[i]
        metrics[f"Specificity ({label})"] = specificity[i]
        metrics[f"G-Score ({label})"] = g_score[i]
    
    return metrics, cm, tpr, g_score

# ---------- PROCESS FILES ----------
rows = []
# hate_type_labels = ["Abusive", "Sexism", "Religious", "Political", "Profane", "None"]
# severity_labels = ["Little to None", "Mild", "Severe"]
# to_whom_labels = ["Individuals", "Organizations", "Communities", "Society", "None"]

# ---------- CORRECT LABELS FROM THE ACTUAL DATASET ----------
hate_type_labels = [
    "Abusive",
    "Political Hate",      # ← was missing "Hate"
    "Profane",
    "Religious Hate",      # ← was missing "Hate"
    "Sexism",
    "None"
]

severity_labels = [
    "Little to None",
    "Mild",
    "Severe"
]

to_whom_labels = [
    "Individual",          # ← singular, not "Individuals"
    "Organization",       # ← singular
    "Community",           # ← singular
    "Society",
    "None"
]

# # Optional: sort alphabetically for consistency (recommended)
# hate_type_labels = sorted(hate_type_labels)
# severity_labels = sorted(severity_labels)
# to_whom_labels = sorted(to_whom_labels)

for file in os.listdir(extract_path):
    if file.endswith(".tsv"):
        df = pd.read_csv(os.path.join(extract_path, file), sep="\t")
        df = df[df["id"].isin(gt_dict)]
        
        y_true = df["id"].map(lambda x: gt_dict[x])
        y_pred = df[["hate_type", "hate_severity", "to_whom"]].values
        
        # Initialize metrics for this file
        file_metrics = {"File name": file}
        
        # Compute combined metrics first
        h_acc = accuracy_score([x[0] for x in y_true], [x[0] for x in y_pred])
        s_acc = accuracy_score([x[1] for x in y_true], [x[1] for x in y_pred])
        w_acc = accuracy_score([x[2] for x in y_true], [x[2] for x in y_pred])
        file_metrics["Combined_Accuracy"] = (h_acc + s_acc + w_acc) / 3
        
        h_precision = precision_score([x[0] for x in y_true], [x[0] for x in y_pred], average='weighted', zero_division=0)
        s_precision = precision_score([x[1] for x in y_true], [x[1] for x in y_pred], average='weighted', zero_division=0)
        w_precision = precision_score([x[2] for x in y_true], [x[2] for x in y_pred], average='weighted', zero_division=0)
        file_metrics["Combined_Precision"] = (h_precision + s_precision + w_precision) / 3
        
        h_recall = recall_score([x[0] for x in y_true], [x[0] for x in y_pred], average='weighted', zero_division=0)
        s_recall = recall_score([x[1] for x in y_true], [x[1] for x in y_pred], average='weighted', zero_division=0)
        w_recall = recall_score([x[2] for x in y_true], [x[2] for x in y_pred], average='weighted', zero_division=0)
        file_metrics["Combined_Recall"] = (h_recall + s_recall + w_recall) / 3
        
        h_f1 = f1_score([x[0] for x in y_true], [x[0] for x in y_pred], average='micro', zero_division=0)
        s_f1 = f1_score([x[1] for x in y_true], [x[1] for x in y_pred], average='micro', zero_division=0)
        w_f1 = f1_score([x[2] for x in y_true], [x[2] for x in y_pred], average='micro', zero_division=0)
        file_metrics["Combined_Micro F1-Score"] = (h_f1 + s_f1 + w_f1) / 3
        
        # Initialize lists to collect TPR and G-Score for combined metrics
        h_tpr, h_g_score = [], []
        s_tpr, s_g_score = [], []
        w_tpr, w_g_score = [], []
        
        # Process each task for class-wise metrics
        for task_idx, task_name, labels in [
            (0, "hate_type", hate_type_labels),
            (1, "hate_severity", severity_labels),
            (2, "to_whom", to_whom_labels)
        ]:
            y_true_task = [x[task_idx] for x in y_true]
            y_pred_task = [x[task_idx] for x in y_pred]
            
            metrics, cm, tpr, g_score = calculate_metrics(y_true_task, y_pred_task, labels)
            
            # Store TPR and G-Score for combined metrics
            if task_name == "hate_type":
                h_tpr, h_g_score = tpr, g_score
            elif task_name == "hate_severity":
                s_tpr, s_g_score = tpr, g_score
            elif task_name == "to_whom":
                w_tpr, w_g_score = tpr, g_score
            
            # Add task-specific prefix to metrics
            for key, value in metrics.items():
                file_metrics[f"{task_name}_{key}"] = value
            
            # ---- Plot Enhanced Confusion Matrix ----
            plt.figure(figsize=(10, 10))
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_percent = np.round(cm_norm * 100, 1)
            cm_percent = np.where(np.isnan(cm_percent), 0.0, cm_percent)
            
            plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Greens)
            # plt.title(f"Confusion Matrix - {task_name} ({file})", fontsize=22, fontname='Times New Roman')
            plt.title(f"Confusion Matrix", fontsize=22, fontname='Times New Roman')

            # Set x-axis rotation only for hate_type
            x_rotation = 0 if task_name == "hate_type" else 0
            x_ha = 'center' if task_name == "hate_type" else 'center'
            plt.xticks(np.arange(len(labels)), labels, fontsize=18, fontname='Times New Roman', rotation=x_rotation, ha=x_ha)
            plt.yticks(np.arange(len(labels)), labels, fontsize=18, fontname='Times New Roman', rotation=90, va='center')
            
            # Annotate counts and percentages
            thresh = cm_norm.max() * 0.75 if cm_norm.max() > 0 else 1
            for i, j in np.ndindex(cm.shape):
                text_color = "white" if cm_norm[i, j] > thresh else "black"
                value = f"{cm[i, j]}\n({cm_percent[i, j]}%)"
                plt.text(j, i, value, ha="center", va="center", color=text_color,
                         fontsize=26, fontname='Times New Roman')
            
            plt.ylabel('True Label', fontsize=22, fontname='Times New Roman')
            plt.xlabel('Predicted Label', fontsize=22, fontname='Times New Roman')
            
            # Class-wise Metrics and Support Below
            supports = [np.sum(np.array(y_true_task) == label) for label in labels]
            prec = precision_score(y_true_task, y_pred_task, average=None, labels=labels, zero_division=0)
            rec = recall_score(y_true_task, y_pred_task, average=None, labels=labels, zero_division=0)
            f1s = f1_score(y_true_task, y_pred_task, average=None, labels=labels, zero_division=0)
            
            metrics_text = "\n".join(
                f"{label} → Support: {supports[i]}, Pr: {prec[i]:.2f}, Re: {rec[i]:.2f}, F1: {f1s[i]:.2f}"
                for i, label in enumerate(labels)
            )
            plt.gcf().text(0.5, -0.14, metrics_text, ha='center', va='bottom',
                           fontsize=16, fontname='Times New Roman')
            
            plt.tight_layout()
            cm_path = os.path.join(confusion_folder, f"{os.path.splitext(file)[0]}_{task_name}_cm.png")
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Average the mean G-Score and TPR across tasks
        h_g_score_avg = np.mean(h_g_score) if h_g_score else 0
        s_g_score_avg = np.mean(s_g_score) if s_g_score else 0
        w_g_score_avg = np.mean(w_g_score) if w_g_score else 0
        file_metrics["Combined_G-Score"] = (h_g_score_avg + s_g_score_avg + w_g_score_avg) / 3
        
        h_tpr_avg = np.mean(h_tpr) if h_tpr else 0
        s_tpr_avg = np.mean(s_tpr) if s_tpr else 0
        w_tpr_avg = np.mean(w_tpr) if w_tpr else 0
        file_metrics["Combined_TPR"] = (h_tpr_avg + s_tpr_avg + w_tpr_avg) / 3
        
        rows.append(file_metrics)

# ---------- SAVE METRICS ----------
metrics_df = pd.DataFrame(rows)
cols = [
    "File name",
    "Combined_Accuracy", "Combined_Precision", "Combined_Recall", "Combined_Micro F1-Score",
    "Combined_G-Score", "Combined_TPR"
] + [c for c in metrics_df.columns if c not in [
    "File name", "Combined_Accuracy", "Combined_Precision", "Combined_Recall",
    "Combined_Micro F1-Score", "Combined_G-Score", "Combined_TPR"
]]
metrics_df = metrics_df[cols]
metrics_df.to_excel(output_xlsx, index=False)

print(f"Metrics saved to: {output_xlsx}")
print(f"Confusion matrices saved in: {confusion_folder}")