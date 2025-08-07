The purpose of this model is to automatically classify event descriptions into categories based on their text content. This helps organize event data, improve search and recommendations, save manual labeling effort, and gain insights into event trends.



**Features:**

* Extracts TF-IDF features from cleaned text data



* Trains classification models (Logistic Regression by default)



* Evaluates model performance with accuracy, precision, recall, F1-score, and confusion matrix visualization



* Makes predictions on new/unseen events to demonstrate usage



**Requirements**

* Python 3.7+



**Required libraries:**

* **pandas**
* **scikit-learn**
* **matplotlib**
* **seaborn**
* **numpy**

Install dependencies with:

* pip install pandas scikit-learn matplotlib seaborn numpy

 

**Usage Run the analyzer (command-line version)**

python prediction.py



Example Output:

Dataset loaded: 1000 samples

Label distribution:

Wedding     300

Corporate   350

Birthday    350

...



TF-IDF feature matrix shape: Train (800, 5000), Test (200, 5000)



Accuracy: 0.85

Macro F1 Score: 0.84

Weighted F1 Score: 0.85



Classification Report:

&nbsp;              precision    recall  f1-score   support

Birthday       0.88       0.82      0.85       70

Corporate      0.83       0.87      0.85       70

Wedding        0.85       0.86      0.85       60



\[Confusion matrix Heatmap displays]



Sample Predictions:

Description: "Autumn wedding ceremony garden"

True Label: Wedding

Predicted Category: Wedding

---

Description: "Corporate team meeting event"

True Label: Corporate

Predicted Category: Corporate

---

...







