# 🌿 AQI-Predication-Flask

A **Flask web application** that predicts the **Air Quality Index (AQI)** based on pollutant data like PM2.5, PM10, CO, NO₂, and more.  
It leverages a pre-trained model or dataset to forecast AQI and helps users understand pollution trends in their area.

---

## ✨ Features

- Predict AQI based on user input for pollutants  
- Interactive and user-friendly web interface  
- Displays predicted AQI values clearly  
- Optional visualization of results using templates

---

## 🗂 Project Structure

AQI-Predication-Flask/
│
├─ app.py # Main Flask application
├─ requirements.txt # Python dependencies
├─ Procfile # Deployment file for Render (optional)
├─ templates/ # HTML templates
│ ├─ index.html
│ └─ layout.html
├─ static/ # CSS, JS, images
│ ├─ style.css
│ └─ script.js
└─ .gitignore

yaml
Copy code
> Adjust folder names if yours differ. Ensure `app.py` (or your main file), templates, static folder, and `requirements.txt` are present.

---

## ⚙️ Prerequisites

- Python 3.8+  
- Flask (and other dependencies listed in `requirements.txt`)  
- Optional: pre-trained model or dataset if used in your project

---

## 💻 Installation — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Kartavyajoshi/AQI-Predication-Flask.git
cd AQI-Predication-Flask

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate             # mac / Linux
# On Windows (PowerShell): .\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
🚀 Running Locally
If your main Flask file is app.py:

bash
Copy code
# mac / Linux
export FLASK_APP=app.py
export FLASK_ENV=development
flask run

# Windows (cmd)
set FLASK_APP=app.py
set FLASK_ENV=development
flask run
Open your browser at http://127.0.0.1:5000 to view the app.

Alternatively, run with Gunicorn (simulates production):

bash
Copy code
gunicorn app:app
Replace app:app with your main file and Flask app variable if different.

🌐 Deployment — Free Hosting with Render
Deploy your app for free on Render:

Go to Render.com and sign up with GitHub.

Click New → Web Service, select the AQI-Predication-Flask repository.

Configure settings:

Environment: Python

Build command: pip install -r requirements.txt

Start command: gunicorn app:app (replace if your main file differs)

Click Create Web Service.

After a few minutes, Render provides a public URL, e.g., https://yourapp.onrender.com.

Pushing changes to GitHub can auto-redeploy the app.

📦 Dependencies
All dependencies are listed in requirements.txt.
Ensure they are installed and up-to-date:

bash
Copy code
pip install --upgrade pip
pip install -r requirements.txt
🤝 Contributing
Fork the repository

Create a new branch: git checkout -b feature-name

Commit your changes: git commit -m "Add feature"

Push: git push origin feature-name

Open a Pull Request

📄 License
Specify your license here (e.g., MIT, Apache 2.0, GPL).
If none, you can add a LICENSE file or note “All rights reserved”.
