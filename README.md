# Banknote Authentication

A small supervised learning project: decide whether a banknote is genuine or forged from four numbers extracted from a photo of it. The model is a logistic regression with feature scaling, built with scikit-learn.

## Data

The Banknote Authentication dataset (originally from the UCI Machine Learning Repository, downloaded via Kaggle) has about 1,370 banknotes. Each note is described by four values calculated from an image of it using a wavelet transform:

- **variance**: how spread out the pixel values are
- **skewness**: how lopsided the pixel values are
- **kurtosis** (spelled `curtosis` in the file): how extreme the unusual pixels are
- **entropy**: how random the image is

The label is `class`: 0 for genuine, 1 for forged.

## Results

The data was split 80/20 with stratification, so both sets keep the same genuine/forged proportions. All scores below are on the 275 notes in the test set, which the model never saw during training.

| Model | Test accuracy |
|---|---|
| Always predict the more common class (baseline) | about 56% |
| Logistic regression with scaling | 97.1% (267 of 275 correct) |

Training accuracy was 98.5%, close to the test score, so the model is not badly overfitting.

| | Predicted genuine | Predicted forged |
|---|---|---|
| **Actually genuine** | 145 | 8 |
| **Actually forged** | 0 | 122 |

All 122 forged notes in the test set were caught, and 8 genuine notes were wrongly flagged as forged. For this kind of task, missing a forgery is the costlier mistake, so errors in this direction are the safer ones.

Model weights (on scaled features): skewness (-4.81), variance (-4.67) and kurtosis (-4.40) pushed strongly toward "genuine" when high, while entropy (+0.26) had almost no influence.

## How it works

1. Load the data and check class balance, missing values and duplicate rows.
2. Split into training and test sets with `stratify=y`.
3. Build a pipeline: `StandardScaler` followed by `LogisticRegression`. The scaler learns its mean and standard deviation from the training data only, which avoids leaking information from the test set.
4. Evaluate with accuracy, a confusion matrix, precision and recall.
5. Save the fitted pipeline with joblib.

## Use the saved model

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python predict.py <variance> <skewness> <curtosis> <entropy>
```

Or from Python:

```python
from predict import predict

predict({"variance": 0.0, "skewness": 0.0, "curtosis": 0.0, "entropy": 0.0})
# returns {"label": "genuine" or "forged", "probability_forged": 0.0 to 1.0}
```

(The numbers above are placeholders. Use real values from the dataset.)

## Limitations

- The test set has only 275 notes, so one extra mistake changes the accuracy by about 0.4 percentage points. Treat the scores as estimates.
- The inputs are numbers already extracted from images. The model cannot take a photo directly.
- This is a learning project on one public dataset, not a tool for real cash handling.

## Credits

Data: Banknote Authentication dataset, UCI Machine Learning Repository, downloaded via Kaggle. Check the dataset page for its license terms before reusing the data.