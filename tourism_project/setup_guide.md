# 🚀 Quick Setup Guide - Tourism MLOps Project

This guide will help you set up and run the complete MLOps pipeline from scratch.

## 📋 Prerequisites Checklist

- [ ] Python 3.9 or higher installed
- [ ] Git installed
- [ ] GitHub account created
- [ ] Hugging Face account created
- [ ] Basic knowledge of terminal/command line

## 🎯 Step-by-Step Setup

### Step 1: Create GitHub Repository

```bash
# Create new repository on GitHub
# Name: tourism-mlops-project
# Make it public
# Initialize with README (optional)
```

### Step 2: Set Up Local Environment

```bash
# Clone your repository
git clone https://github.com/YourUsername/tourism-mlops-project.git
cd tourism-mlops-project

# Create the project structure
mkdir -p tourism_project/{data,model_building,deployment/.streamlit,hosting,.github/workflows}

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Add Your Data

```bash
# Place your tourism.xlsx file in the data folder
cp path/to/your/tourism.xlsx tourism_project/data/
```

### Step 4: Get Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it: `tourism-mlops-token`
4. Type: `Write`
5. Click "Generate"
6. **Copy the token** (you'll need it in the next step)

### Step 5: Configure GitHub Secrets

1. Go to your GitHub repository
2. Click: **Settings** → **Secrets and variables** → **Actions**
3. Click: **New repository secret**
4. Name: `HF_TOKEN`
5. Value: Paste your Hugging Face token
6. Click: **Add secret**

### Step 6: Update Configuration Files

Edit the following files and replace `TheHumanAgent` with your Hugging Face username:

#### `model_building/data_register.py`
```python
# Line ~60
repo_id = "YourHFUsername/tourism-data"
```

#### `model_building/train.py`
```python
# Line ~120
repo_id = "YourHFUsername/tour_pkg_pred_model"
```

#### `hosting/hosting.py`
```python
# Line ~12
REPO_ID = "YourHFUsername/tour-pkg-predictor"
```

#### `deployment/app.py`
```python
# Line ~54
repo_id="YourHFUsername/tour_pkg_pred_model"
```

### Step 7: Add All Files to Repository

```bash
# Add all project files
git add .

# Commit changes
git commit -m "Initial commit: MLOps pipeline setup"

# Push to GitHub
git push origin main
```

### Step 8: Verify Pipeline Execution

1. Go to your GitHub repository
2. Click on **Actions** tab
3. You should see the workflow running
4. Click on the workflow to see progress
5. Wait for all jobs to complete (approximately 10-15 minutes)

### Step 9: Check Deployments

After pipeline completes successfully, verify:

#### ✅ Dataset on Hugging Face
```
https://huggingface.co/datasets/YourUsername/tourism-data
```

#### ✅ Model on Hugging Face
```
https://huggingface.co/YourUsername/tour_pkg_pred_model
```

#### ✅ Streamlit App on Hugging Face
```
https://huggingface.co/spaces/YourUsername/tour-pkg-predictor
```

## 🧪 Local Testing (Optional)

### Test Data Registration

```bash
cd tourism_project/model_building
export HF_TOKEN="your_token_here"
python data_register.py
```

### Test Data Preparation

```bash
python prep.py
```

### Test Model Training

```bash
# Start MLflow server
mlflow ui --host 0.0.0.0 --port 5000 &

# Train model
python train.py

# View MLflow UI
# Open browser: http://localhost:5000
```

### Test Streamlit App Locally

```bash
cd ../deployment
pip install -r requirements.txt
streamlit run app.py
```

### Test Deployment Script

```bash
cd ../hosting
python hosting.py
```

## 🔧 Troubleshooting

### Issue: "HF_TOKEN not found"

**Solution:**
```bash
# Set environment variable
export HF_TOKEN="your_token_here"  # Linux/Mac
set HF_TOKEN="your_token_here"     # Windows CMD
$env:HF_TOKEN="your_token_here"    # Windows PowerShell
```

### Issue: "Repository not found"

**Solution:**
- Verify you've updated all configuration files with your username
- Check that HF_TOKEN has write permissions
- Ensure you're logged in to Hugging Face: `huggingface-cli login`

### Issue: "Module not found"

**Solution:**
```bash
# Install missing dependencies
pip install -r tourism_project/deployment/requirements.txt

# Or install individually
pip install pandas numpy scikit-learn xgboost mlflow huggingface-hub streamlit plotly
```

### Issue: "GitHub Actions workflow fails"

**Solution:**
1. Check the Actions tab for error details
2. Verify HF_TOKEN is set in GitHub Secrets
3. Check that all file paths are correct
4. Review the logs for specific error messages

### Issue: "Streamlit app not loading model"

**Solution:**
1. Verify model is uploaded to HF: Check model hub URL
2. Ensure repo_id in app.py matches your HF username
3. Check that model file name is correct: `final_tour_pkg_pred_model_v1.joblib`

## 📊 Monitoring Your Pipeline

### View GitHub Actions

```
https://github.com/YourUsername/tourism-mlops-project/actions
```

### View MLflow Experiments

```bash
# After running training locally
mlflow ui --host 0.0.0.0 --port 5000
# Open: http://localhost:5000
```

### View Hugging Face Spaces Logs

1. Go to your Space URL
2. Click on **Logs** tab
3. Monitor build and runtime logs

## 🎉 Success Criteria

Your setup is complete when:

- ✅ GitHub Actions workflow runs successfully
- ✅ All 4 jobs complete (green checkmarks)
- ✅ Dataset appears on HF Dataset Hub
- ✅ Model appears on HF Model Hub
- ✅ Streamlit app is live on HF Spaces
- ✅ App can make predictions
- ✅ You can download prediction reports

## 📞 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review GitHub Actions logs
3. Check Hugging Face Space logs
4. Verify all configuration files
5. Ensure all dependencies are installed

## 🎓 Next Steps

After successful setup:

1. **Test the application**: Make some predictions
2. **Review MLflow logs**: Check experiment tracking
3. **Customize the app**: Modify styling or add features
4. **Share your work**: Add screenshots to README
5. **Document learnings**: Update project documentation

## 📝 Checklist for Submission

- [ ] GitHub repository is public
- [ ] All code is pushed to main branch
- [ ] GitHub Actions workflow executed successfully
- [ ] Dataset uploaded to HF Dataset Hub
- [ ] Model uploaded to HF Model Hub
- [ ] Streamlit app deployed on HF Spaces
- [ ] README.md has project documentation
- [ ] Screenshots added (folder structure, workflow, app)
- [ ] All links work correctly

## 🏆 Expected Deliverables

1. **GitHub Repository**
   - Clean folder structure
   - Executed workflow (screenshot)
   - Working pipeline.yml

2. **Hugging Face**
   - Dataset space
   - Model hub space
   - Live Streamlit app
   - App screenshot

3. **Documentation**
   - Comprehensive README
   - Code comments
   - Setup instructions

---

**Estimated Setup Time**: 30-45 minutes  
**Pipeline Execution Time**: 10-15 minutes  
**Total Time**: ~1 hour

Good luck with your MLOps project! 🚀
