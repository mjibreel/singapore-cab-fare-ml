# GitHub Setup Guide

## Steps to Upload Your Project to GitHub

### 1. Create a New Repository on GitHub
1. Go to [github.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name it: `singapore-taxi-fare-prediction`
5. Make it **Public** (for school project)
6. Don't initialize with README (we already have one)
7. Click "Create repository"

### 2. Initialize Git in Your Local Project
Open Terminal/Command Prompt in your project folder and run:

```bash
# Navigate to your project directory
cd /Users/mohammadjibril/Desktop/ml/singapore-taxi-fare-prediction

# Initialize git repository
git init

# Add all files
git add .

# Make first commit
git commit -m "Initial commit: Singapore Taxi Fare Prediction System"

# Add your GitHub repository as remote origin
git remote add origin https://github.com/YOUR_USERNAME/singapore-taxi-fare-prediction.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 3. Replace YOUR_USERNAME
In the command above, replace `YOUR_USERNAME` with your actual GitHub username.

### 4. Verify Upload
- Go to your GitHub repository
- You should see all your files uploaded
- The README.md will automatically display on the main page

### 5. Optional: Enable GitHub Pages (for web demo)
1. Go to your repository Settings
2. Scroll down to "Pages" section
3. Source: "Deploy from a branch"
4. Branch: "main" / "/ (root)"
5. Click "Save"

### 6. Future Updates
When you make changes to your code:

```bash
git add .
git commit -m "Description of your changes"
git push
```

## Project Structure on GitHub
Your repository will contain:
- `README.md` - Project overview and instructions
- `streamlit_app.py` - Main Streamlit web application
- `main.py` - Command-line version
- `fare_calculator.py` - Fare calculation logic
- `haversine.py` - Distance calculation utilities
- `requirements.txt` - Python dependencies
- `WINDOWS_SETUP.md` - Windows setup guide
- `Proposal_Report.md` - Your project report
- `prd.md` - Product requirements document

## Benefits of GitHub
✅ **Portfolio**: Show your coding skills to teachers/employers
✅ **Collaboration**: Easy to share and work with others
✅ **Version Control**: Track all your changes
✅ **Professional**: Industry standard for code sharing
✅ **Free Hosting**: GitHub Pages for web demos

## Need Help?
If you encounter any issues:
1. Check the error message in Terminal
2. Make sure you're in the correct directory
3. Verify your GitHub username is correct
4. Ensure you have Git installed on your computer
