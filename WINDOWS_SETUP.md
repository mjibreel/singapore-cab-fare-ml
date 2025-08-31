# 🪟 Windows Setup Guide for Singapore Taxi Fare Prediction

This guide will help you set up and run the Singapore Taxi Fare Prediction app on your Windows laptop.

## 📋 Prerequisites

### 1. Python Installation
- **Download Python 3.9 or 3.10** from [python.org](https://www.python.org/downloads/)
- **Important**: Check "Add Python to PATH" during installation
- **Verify installation**: Open Command Prompt and type `python --version`

### 2. Git (Optional but Recommended)
- Download from [git-scm.com](https://git-scm.com/download/win)
- Install with default settings

## 🚀 Installation Steps

### Step 1: Download the Project
```cmd
# Option A: Using Git (recommended)
git clone <your-repository-url>
cd singapore-taxi-fare-prediction

# Option B: Download ZIP file
# 1. Download the ZIP file from your repository
# 2. Extract it to a folder (e.g., C:\Projects\singapore-taxi-fare-prediction)
# 3. Open Command Prompt and navigate to the folder
cd C:\Projects\singapore-taxi-fare-prediction
```

### Step 2: Create Virtual Environment
```cmd
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# You should see (venv) at the beginning of your command prompt
```

### Step 3: Install Dependencies
```cmd
# Upgrade pip first
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Step 4: Run the App
```cmd
# Make sure your virtual environment is activated (venv)
streamlit run streamlit_app.py
```

## 🌐 Accessing the App

After running the command, you should see:
```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

1. **Open your web browser**
2. **Go to**: `http://localhost:8501`
3. **Your app will load!** 🎉

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: "python is not recognized"
**Solution**: Python is not in PATH
```cmd
# Reinstall Python and check "Add Python to PATH"
# Or manually add Python to PATH in System Environment Variables
```

#### Issue 2: "pip is not recognized"
**Solution**: Use python -m pip instead
```cmd
python -m pip install -r requirements.txt
```

#### Issue 3: "Microsoft Visual C++ 14.0 is required"
**Solution**: Install Visual Studio Build Tools
```cmd
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Or install pre-compiled wheels
pip install --only-binary=all -r requirements.txt
```

#### Issue 4: "Permission denied" when creating venv
**Solution**: Run Command Prompt as Administrator
```cmd
# Right-click Command Prompt → "Run as administrator"
```

#### Issue 5: Streamlit won't start
**Solution**: Check if port 8501 is available
```cmd
# Try a different port
streamlit run streamlit_app.py --server.port 8502
```

## 📁 Project Structure on Windows

```
C:\Projects\singapore-taxi-fare-prediction\
├── streamlit_app.py          # Main web application
├── main.py                   # Command-line version
├── haversine.py             # Distance calculations
├── fare_calculator.py       # Fare calculation logic
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── WINDOWS_SETUP.md         # This file
└── venv\                    # Virtual environment (created during setup)
```

## 🎯 Running Different Versions

### Option 1: Streamlit Web App (Recommended)
```cmd
streamlit run streamlit_app.py
```
- **Beautiful web interface**
- **Interactive elements**
- **Perfect for presentations**

### Option 2: Command Line App
```cmd
python main.py
```
- **Simple text interface**
- **Good for testing**
- **No web browser needed**

## 🚀 Quick Start Commands

```cmd
# Navigate to project folder
cd C:\Projects\singapore-taxi-fare-prediction

# Activate virtual environment
venv\Scripts\activate

# Install dependencies (first time only)
pip install -r requirements.txt

# Run the web app
streamlit run streamlit_app.py

# Or run command line version
python main.py
```

## 💡 Tips for Windows Users

1. **Use Command Prompt or PowerShell** (avoid Git Bash for pip installs)
2. **Keep your virtual environment activated** while working
3. **Use `Ctrl+C`** to stop Streamlit when done
4. **Check Windows Defender** if you have firewall issues
5. **Use Windows Terminal** (available from Microsoft Store) for better experience

## 🆘 Getting Help

If you encounter issues:

1. **Check the error message** carefully
2. **Verify Python version**: `python --version`
3. **Verify pip version**: `pip --version`
4. **Check if virtual environment is activated**: Should see `(venv)` in prompt
5. **Try running as Administrator** if you get permission errors

## 🎉 Success!

Once everything is working, you'll have:
- ✅ A beautiful web application running on your Windows laptop
- ✅ Professional-looking interface for your school project
- ✅ Interactive taxi fare prediction system
- ✅ Two different pricing models to compare

Your friends will be impressed by your professional web application! 🚕✨

---

**Need more help?** Check the main README.md file or create an issue in your repository.

