# 🧩 Bangla Multi-task Hate Speech Identification — Subtask 1C  
### System Reproducibility Repository (BLP Workshop @ IJCNLP-AACL 2025)

This repository contains the implementation, experimental setup, and utilities for our submitted system to the **Bangla Multi-task Hate Speech Identification Shared Task**, organized as part of the **Bangla Language Processing (BLP) Workshop @ IJCNLP-AACL 2025**.

- To cite our paper, please use following ```bibtex``` command.


    ```bibtex
    @inproceedings{hossan-etal-2025-cuet-nlp-zenith,
        title = "{CUET}-{NLP}{\_}{Z}enith at {BLP}-2025 Task 1: A Multi-Task Ensemble Approach for Detecting Hate Speech in {B}engali {Y}ou{T}ube Comments",
        author = "Hossan, Md. Refaj  and
          Ahmed, Kawsar  and
          Hoque, Mohammed Moshiul",
        editor = "Alam, Firoj  and
          Kar, Sudipta  and
          Chowdhury, Shammur Absar  and
          Hassan, Naeemul  and
          Prince, Enamul Hoque  and
          Tasnim, Mohiuddin  and
          Rony, Md Rashad Al Hasan  and
          Rahman, Md Tahmid Rahman",
        booktitle = "Proceedings of the Second Workshop on Bangla Language Processing (BLP-2025)",
        month = dec,
        year = "2025",
        address = "Mumbai, India",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2025.banglalp-1.38/",
        pages = "443--452",
        ISBN = "979-8-89176-314-2",
        abstract = "Hate speech on social media platforms, particularly in low-resource languages like Bengali, poses a significant challenge due to its nuanced nature and the need to understand its type, severity, and targeted group. To address this, the Bangla Multi-task Hate Speech Identification Shared Task at BLP 2025 adopts a multi-task learning framework that requires systems to classify Bangla YouTube comments across three subtasks simultaneously: type of hate, severity, and targeted group. To tackle these challenges, this work presents BanTriX, a transformer ensemble method that leverages BanglaBERT-I, XLM-R, and BanglaBERT-II. Evaluation results show that the BanTriX, optimized with cross-entropy loss, achieves the highest weighted micro F1-score of 73.78{\%} in Subtask 1C, securing our team 2nd place in the shared task."
    }
    ```

## 🧠 Task Overview

**Objective:**  
The shared task aims to identify and analyze **hate speech in Bangla**, focusing on multi-task classification to predict:
- **Hate Type**
- **Hate Severity**
- **Target Group**

In **Subtask 1C**, models are required to jointly predict these three aspects for a given Bangla YouTube comment.  
This setup reflects real-world scenarios where understanding hate speech involves more than binary classification.

**Data Format:**

| id | text | hate_type | hate_severity | to_whom |
|----|------|------------|----------------|----------|
| 4***3 | রাজনৈতিক দলের সন্ত্রাসী কবে ধরবেন এই সাহস আপনাদের নাই | Political Hate | Little to None | Organization |

---

## 📁 Repository Structure

```

Notebooks/
└── 1c-proposed-architecture.ipynb      # Main model architecture and training notebook

Results/
├── confusion_matrices/                 # Visualizations of confusion matrices for all subtasks
└── summary_metrics.xlsx                # Summary of performance metrics (accuracy, F1, etc.)

Utils/
├── predictions_extracted/              # Extracted model predictions for analysis
├── dataset_statistics.ipynb            # Dataset-level statistics and EDA
├── scorer.py                           # Official scoring script (weighted micro-F1 score)
└── truth.tsv                           # Ground truth labels for evaluation

Prediction.zip                           # Model prediction files for test sets
README.md                                # Project documentation (this file)

```

---

## ⚙️ Methodology

After publishing the paper, this section will be updated soon with detailed information on the model architecture, data preprocessing techniques, and classification approaches used.

## 🚀 Running the Code

### **1. Environment Setup**
```bash
pip install -r requirements.txt
````

### **2. Training**

```bash
jupyter notebook Notebooks/1c-proposed-architecture.ipynb
```

### **3. Evaluation**

```bash
python Utils/scorer.py --pred Prediction.zip --truth Utils/truth.tsv
```

### **4. Reproducing Metrics**

* Confusion matrices and F1-scores can be found in the `Results` folder.

---

## 📬 Contact

For queries or collaborations, please contact:
**Md Refaj Hossan**, Research Assistant, CUET NLP Lab, Department of CSE, CUET

📧 [[mdrefajhossan@gmail.com](mailto:mdrefajhossan@gmail.com)]

🌐 [[Website](https://refaj-hossan.vercel.app/)]

---

## 🪪 License

This repository is released under the **MIT License**. Please see the LICENSE file for details.

