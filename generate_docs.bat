@echo off
title SOLIX Document Generator
echo =======================================================
echo     SOLIX Virtual Data Engineer Document Generator
echo =======================================================
echo.
echo Running Python documentation generator...
python docs_scripts/run_all_v2.py
echo.
echo =======================================================
echo Done! Please check docs_output directory.
echo =======================================================
pause
