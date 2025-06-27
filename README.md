# 🧠 ML Dataset Analyzer

**AI-powered web app to detect ML tasks, suggest models, train & predict — no code needed.**

A smart and interactive Flask-based web application that allows users to upload CSV datasets, auto-detect the machine learning task, get LLM-generated model suggestions, train multiple models, compare performance, predict on new data, and download results — all via a clean Bootstrap UI.

---

## 🚀 Features

| Category | Description |
|----------|-------------|
| 📁 File Upload | Upload new datasets or reuse previous ones |
| 🎯 Target Selection | Manually select target column from dropdown |
| 🤖 LLM Suggestions | Uses OpenRouter (Gemma-3-27B) to recommend ML models |
| 📝 Prompt Editing | View and edit the prompt sent to the LLM |
| 🔁 Re-run Detection | Retry model suggestions after tweaking the prompt |
| 🧠 Task Detection | Automatically classifies as Regression, Classification, or Time Series |
| 🔐 LLM Fallback | Shows default model advice if API fails |
| 📊 Model Training | Train Logistic Regression, SVM, XGBoost, Random Forest, and more |
| 📉 Metric Reporting | Accuracy, Precision, Recall, F1, AUC, R², RMSE |
| ⚠️ Performance Alert | Warns if Accuracy < 60% or R² < 0.5 |
| 💾 Save Best Model | Automatically saves the best model as `best_model.pkl` |
| 📥 Predict on New Data | Upload new CSV and get predictions |
| 📤 Download Results | Download results as timestamped CSV |
| 🎨 Bootstrap UI | Clean, responsive interface |
| 🔢 Token Count | Shows token usage before LLM call |

---

## 🧪 Supported ML Models

- Logistic Regression
- Decision Tree
- Random Forest
- Linear Regression
- Support Vector Machine
- K-Nearest Neighbors
- Naive Bayes
- XGBoost (Classifier & Regressor)

---

## 🛠 Technologies Used

- Python (Flask)
- HTML, CSS (Bootstrap 5)
- OpenRouter API (Gemma-3-27B)
- scikit-learn, XGBoost, SHAP (optional)
- Pandas, NumPy, Matplotlib
- Joblib (for saving models)

---

## 📦 Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/ml-dataset-analyzer.git
cd ml-dataset-analyzer

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
# Create a .env file with this content:
OPENROUTER_API_KEY=your_openrouter_key_here

# 5. Run the app
python app.py

## 🚧 Challenges Faced

### 🔐 1. LLM API Key Handling with OpenRouter

One of the major challenges was integrating **OpenRouter's API** to dynamically suggest the best machine learning model for any uploaded dataset. 

Initially, attempts to use the `BASE_URL` for making requests to non-OpenAI models failed because:

- OpenRouter does **not support direct base URL calls** (`https://openrouter.ai/v1`) for models like `mistral`, `anthropic`, etc.
- Instead, these models must be accessed by including the **model name and API key in the request headers**, as per OpenRouter’s SDK guidelines.

This required adjusting the request logic to correctly format headers and payloads for compatibility with OpenRouter’s routing.

---

### 🐍 2. Python Version Mismatch During Deployment on Render

Another major roadblock was encountered while deploying the Flask app to **Render**.

By default, Render uses **Python 3.13**, which is **incompatible with many popular ML libraries** such as:
- `scikit-learn`
- `shap`
- `xgboost`
- `numpy`

This caused multiple build failures with cryptic dependency resolution errors.

🔧 **Solution:**
- Created a `runtime.txt` file with `python-3.10.13`
- Added an environment variable: `PYTHON_VERSION=3.10.13` in Render’s settings
- Redeployed with **"Clear build cache & deploy latest commit"**

This ensured all dependencies installed successfully and the app deployed without errors.

---

