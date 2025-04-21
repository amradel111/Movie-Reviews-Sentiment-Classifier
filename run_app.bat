@echo off
echo Attempting to start Movie Reviews Sentiment Analyzer...

:: Activate venv if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Using system Python...
)

echo Running python app.py...
echo Any errors will be shown below.

:: Run the python script directly
python app.py

:: Keep window open after script finishes or errors out
echo.
echo Script finished or encountered an error.
echo Check app_debug.log for details.
pause 