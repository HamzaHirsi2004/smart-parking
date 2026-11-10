@echo off
cd /d "%~dp0\.."

start "AI Worker" cmd /k "venv\Scripts\activate && python detection\worker.py --source data\demo.mp4"
start "Customer Website" cmd /k "venv\Scripts\activate && streamlit run customer\app.py --server.port 8501"
start "Admin Dashboard" cmd /k "venv\Scripts\activate && streamlit run admin\app.py --server.port 8502"

echo.
echo  Customer site: http://localhost:8501
echo  Admin dashboard: http://localhost:8502
echo  Close the three windows to stop the system.