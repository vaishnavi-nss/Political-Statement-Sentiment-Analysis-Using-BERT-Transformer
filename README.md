# Political Statement Sentiment Analysis 

This project analyzes the sentiment behind political statements using a fine-tuned BERT transformer model. It classifies each statement into **positive**, **negative**, or **neutral** sentiment categories, helping to understand the tone and public perception of political discourse.

---

## 📌 Table of Contents

* [About the Project](#about-the-project)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Installation](#installation)
* [Usage](#usage)
* [Results](#results)
* [Technologies Used](#technologies-used)
* [Future Work](#future-work)
* [License](#license)

---

## 📖 About the Project

Political communication often influences public opinion, policymaking, and media coverage. This project leverages **BERT (Bidirectional Encoder Representations from Transformers)** to perform **sentiment analysis** on political statements, aiming to:

* Identify the **tone** behind statements (positive/negative/neutral)
* Track **patterns** in political sentiment over time
* Assist researchers or journalists in analyzing speech content objectively

---

## 📂 Dataset

* A dataset of political statements was compiled from various sources including:

  * Debates
  * Interviews
  * Press releases
* Each statement is labeled with sentiment: `positive`, `neutral`, or `negative`

> *Custom preprocessing was applied to clean and tokenize text.*

---

## 🧠 Model Architecture

* **Base model**: `bert-base-uncased` from HuggingFace Transformers
* Fine-tuned with:

  * Classification head (dense layer)
  * CrossEntropy loss
  * AdamW optimizer

---

## ⚙️ Installation

```bash
git clone (https://github.com/vaishnavi-nss/Political-Statement-Sentiment-Analysis-Using-BERT-Transformer)
cd Political-Statement-Sentiment-Analysis-Using-BERT-Transformer
pip install -r requirements.txt
```

**Main libraries**:

* `transformers`
* `torch`
* `scikit-learn`
* `pandas`

---

## 🚀 Usage

### Training

```bash
python train.py --epochs 3 --batch_size 16 --lr 2e-5
```

### Inference

```bash
python predict.py --text "We will bring prosperity to every citizen."
```

---

## 📊 Results

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 87.2% |
| F1 Score  | 85.5% |
| Precision | 86.1% |
| Recall    | 84.7% |

Sample Predictions:

* "The government failed us again." → **Negative**
* "We are proud of our nation’s progress." → **Positive**

---

## 🛠 Technologies Used

* BERT (HuggingFace Transformers)
* PyTorch
* Scikit-learn
* Pandas, NumPy
* Matplotlib/Seaborn (for visualization)

---

## 🔮 Future Work

* Deploy model via Flask/Streamlit for live demo
* Extend to multilingual datasets
* Add context-aware classification (e.g., sarcasm detection)
* Visualize sentiment trends across political parties

---

## 📄 License

MIT License. Feel free to use and contribute.

---

## 🙌 Acknowledgements

* [HuggingFace Transformers](https://huggingface.co/transformers/)
* [BERT Paper](https://arxiv.org/abs/1810.04805)
* Public political datasets from Kaggle and media transcripts

---
