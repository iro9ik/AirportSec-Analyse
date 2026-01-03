# Airport Security Analysis Platform

A professional desktop application for **Airport Security Analysis**, combining  
**Data Analysis, Artificial Intelligence, and Cybersecurity** techniques.

The application is built with **Python + Tkinter** and packaged into a **Windows executable** using **PyInstaller**.

---

## ✈️ Features

### 📊 Data Analysis
- Descriptive statistics
- Correlation analysis
- Principal Component Analysis (PCA / ACP)
- Interactive tables and visualizations

### 🤖 Artificial Intelligence
- K-Means clustering (k = 3 to 7)
- Random Forest classification
- Cluster quality metrics (silhouette score, inertia)
- Prediction of new airport zones

### 🔐 Cybersecurity Analysis
- Anomaly detection using:
  - Isolation Forest
  - Local Outlier Factor (LOF)
- Risk zone identification
- Visual risk mapping
- Automated cybersecurity reports

---

## 🖥️ User Interface
- Modern Tkinter-based UI
- Interactive charts (Matplotlib)
- Responsive layout
- Navigation panels and dashboards

---

## 📁 Project Structure

AirportSec-Analyse/
├── airport.py # Main application
├── airport.spec # PyInstaller build configuration
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore # Git ignore rules


> ⚠️ The folders `build/` and `dist/` are **NOT** tracked in Git  
> The compiled `.exe` is distributed via **GitHub Releases**.

---

## 🚀 How to Run (Development Mode)

### 1️⃣ Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Run the application

python airport.py

📦 Build the Windows Executable (.exe)

Make sure PyInstaller is installed:

pip install pyinstaller

Build using the provided spec file:

pyinstaller airport.spec

The executable will be generated in:

dist/airport.exe

📥 Distribution

    The .exe file is not stored in the Git repository

    Downloadable executables are published under:

👉 GitHub → Releases → Version tag
🧠 Technologies Used

    Python 3.9+

    Tkinter

    NumPy

    Pandas

    Matplotlib

    Seaborn

    Scikit-learn

    SciPy

    PyInstaller

📜 License

This project is provided for educational and analytical purposes.
👤 Author

Ahmed Abounaoum
Airport Security – Data Analysis & AI Project
