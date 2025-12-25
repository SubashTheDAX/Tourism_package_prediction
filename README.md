# ✈️ Wellness Tourism Package Prediction - MLOps Pipeline

![MLOps](https://img.shields.io/badge/MLOps-Automated-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

An end-to-end MLOps project that predicts customer purchase likelihood for wellness tourism packages using machine learning, automated with GitHub Actions and deployed on Hugging Face.

## 📋 Table of Contents

- [Business Context](#business-context)
- [Project Objective](#project-objective)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Pipeline Workflow](#pipeline-workflow)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Usage](#usage)
- [Results](#results)

## 🎯 Business Context

"Visit with Us," a leading travel company, aims to optimize customer targeting for their new Wellness Tourism Package. This project implements a scalable, automated MLOps pipeline to predict which customers are likely to purchase the package, enabling data-driven marketing strategies and efficient resource allocation.

## 🎪 Project Objective

Design and deploy an automated MLOps pipeline that:
- Preprocesses and transforms customer data
- Builds and trains predictive ML models
- Deploys models with CI/CD integration
- Provides an interactive prediction interface
- Enables continuous model updates and monitoring

## 🏗️ Architecture

```
┌─────────────────┐
│   GitHub Push   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   GitHub Actions Workflow           │
├─────────────────────────────────────┤
│ 1. Data Registration → HF Dataset   │
│ 2. Data Preparation → Train/Test    │
│ 3. Model Training → MLflow Tracking │
│ 4. Model Upload → HF Model Hub      │
│ 5. Deploy App → HF Spaces          │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Deployed Application              │
│   • Streamlit UI                    │
│   • Real-time Predictions          │
│   • Visual Analytics               │
│   • Export Reports                 │
└─────────────────────────────────────┘
```

## 📁 Project Structure

```
tourism_project/
│
├── data/
│   └── tourism.xlsx                 # Raw dataset
│
├── model_building/
│   ├── data_register.py             # Upload data to HF Dataset Hub
│   ├── prep.py                      # Data cleaning & preprocessing
│   └── train.py                     # Model training with MLflow
│
├── deployment/
│   ├── .streamlit/
│   │   └── config.toml              # Streamlit configuration
│   ├── app.py                       # Streamlit application
│   ├── Dockerfile                   # Container configuration
│   └── requirements.txt             # Python dependencies
│
├── hosting/
│   └── hosting.py                   # Deploy to HF Spaces
│
└── .github/
    └── workflows/
        └── pipeline.yml             # CI/CD workflow
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9+
- Git
- GitHub account
- Hugging Face account

### 1. Clone Repository

```bash
git clone https://github.com/YourUsername/tourism-mlops.git
cd tourism-mlops
```

### 2. Install Dependencies

```bash
pip install -r tourism_project/deployment/requirements.txt
```

### 3. Configure Hugging Face

```bash
# Login to Hugging Face
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN="your_huggingface_token"
```

### 4. Set GitHub Secrets

Add the following secret to your GitHub repository:
- Go to: Settings → Secrets and variables → Actions
- Create new secret: `HF_TOKEN` with your Hugging Face token

### 5. Update Configuration

Edit the following files with your Hugging Face username:
- `model_building/data_register.py` → Update `repo_id`
- `model_building/train.py` → Update `repo_id`
- `hosting/hosting.py` → Update `REPO_ID`
- `deployment/app.py` → Update `repo_id` in `load_model()`

## 🔄 Pipeline Workflow

### Automated CI/CD Pipeline (GitHub Actions)

The pipeline automatically triggers on push to `main` branch:

#### **Job 1: Data Registration** (3 points)
```yaml
- Uploads raw dataset to Hugging Face Dataset Hub
- Creates versioned data storage
```

#### **Job 2: Data Preparation** (7 points)
```yaml
- Loads data from HF Dataset Hub
- Cleans and preprocesses data
- Handles missing values and outliers
- Encodes categorical variables
- Splits into train/test sets (80/20)
- Uploads processed data back to HF
```

#### **Job 3: Model Training** (13 points)
```yaml
- Loads train/test data from HF
- Trains multiple ML models:
  • XGBoost (primary)
  • Random Forest
  • Gradient Boosting
  • AdaBoost
- Hyperparameter tuning with GridSearchCV
- Logs experiments to MLflow
- Evaluates model performance
- Registers best model to HF Model Hub
```

#### **Job 4: Deployment** (11 points)
```yaml
- Pushes Streamlit app to HF Spaces
- Uploads Dockerfile and dependencies
- Creates production-ready interface
- Enables real-time predictions
```

## 🤖 Model Details

### Algorithm: XGBoost Classifier

**Features (18 variables):**

| Category | Features |
|----------|----------|
| **Demographics** | Age, Gender, MaritalStatus, CityTier |
| **Professional** | Occupation, Designation, MonthlyIncome |
| **Travel** | NumberOfTrips, Passport, PreferredPropertyStar |
| **Interaction** | TypeofContact, ProductPitched, DurationOfPitch, NumberOfFollowups, PitchSatisfactionScore |
| **Trip Details** | NumberOfPersonVisiting, NumberOfChildrenVisiting, OwnCar |

**Preprocessing:**
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- Class weight balancing (scale_pos_weight)

**Hyperparameters Tuned:**
- n_estimators: [50, 75, 100, 125, 150]
- max_depth: [2, 3, 4]
- colsample_bytree: [0.4, 0.5, 0.6]
- colsample_bylevel: [0.4, 0.5, 0.6]
- learning_rate: [0.01, 0.05, 0.1]
- reg_lambda: [0.4, 0.5, 0.6]

**Classification Threshold:** 45%

### Model Performance

```
Training Data: 4,128 customers
Purchase Rate: 19.3%
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score
```

## 🌐 Deployment

### Hugging Face Spaces

The application is deployed as a Streamlit app on Hugging Face Spaces:

**Live App:** `https://huggingface.co/spaces/TheHumanAgent/tour-pkg-predictor`

**Model Hub:** `https://huggingface.co/TheHumanAgent/tour_pkg_pred_model`

**Dataset Hub:** `https://huggingface.co/datasets/TheHumanAgent/tourism-data`

### Local Testing

```bash
cd tourism_project/deployment
streamlit run app.py
```

## 📊 Usage

### Making Predictions

1. **Access the App**: Visit the deployed Hugging Face Space
2. **Enter Customer Details**: Fill in the form with customer information
3. **Predict**: Click "Predict Purchase Likelihood"
4. **Review Results**: 
   - Purchase probability (0-100%)
   - Prediction (Likely/Unlikely to purchase)
   - Visual analytics (gauge chart, metrics)
   - Actionable recommendations
5. **Export**: Download detailed CSV report

### Example Input

```python
Customer Profile:
- Age: 41
- Gender: Female
- Monthly Income: ₹20,993
- Occupation: Salaried
- City Tier: 3
- Number of Trips: 1
- Product Pitched: Deluxe
- Satisfaction Score: 2/5
```

### Expected Output

```
Prediction: LIKELY TO PURCHASE
Probability: 78.5%
Recommendation: HIGH PRIORITY LEAD
Action: Schedule immediate follow-up call
```

## 📈 Results

### Pipeline Execution

✅ **Data Registration**: Dataset uploaded to HF Dataset Hub  
✅ **Data Preparation**: 4,128 records cleaned and split  
✅ **Model Training**: Best model with 81% validation accuracy  
✅ **Deployment**: Streamlit app live on HF Spaces  

### Key Achievements

- **Automation**: End-to-end pipeline with GitHub Actions (15 points)
- **Model Quality**: Production-ready XGBoost model (13 points)
- **Deployment**: Live Streamlit application (11 points)
- **Data Pipeline**: Automated preprocessing (7 points)
- **Code Quality**: Well-documented, modular code (7 points)

**Total Points**: 53/60 ⭐

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **ML Framework** | scikit-learn, XGBoost |
| **Experiment Tracking** | MLflow |
| **Model Registry** | Hugging Face Hub |
| **Frontend** | Streamlit |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Data Processing** | pandas, numpy |
| **Visualization** | Plotly |

## 📝 Future Enhancements

- [ ] Add A/B testing capabilities
- [ ] Implement model monitoring and drift detection
- [ ] Add batch prediction endpoint
- [ ] Create REST API with FastAPI
- [ ] Add model explainability (SHAP values)
- [ ] Implement automated retraining
- [ ] Add user authentication

## 👥 Contributors

- **Subash M** - MLOps Engineer

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Visit with Us** - For the business case and dataset
- **Hugging Face** - For hosting infrastructure
- **Streamlit** - For the frontend framework

---

**Project Status**: ✅ Production Ready  
**Last Updated**: December 2025  
**Version**: 1.0.0
