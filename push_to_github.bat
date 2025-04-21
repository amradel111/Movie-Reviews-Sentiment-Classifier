@echo off
echo Pushing to GitHub repository...

git remote add origin https://github.com/amradel111/imdb-sentiment-classifier.git
git branch -M main
git push -u origin main

echo Done!
pause 