@echo off
title SOLIX Premium Document Compiler
echo =======================================================
echo     SOLIX Premium Word Document Compiler (.docx)
echo =======================================================
echo.
echo Compiling Chapter 1 and Chapter 2...
python scratch/compile_docs.py
echo.
echo =======================================================
echo Done! Please check:
echo SOLIX_Project_Premium_Documentation.docx in this folder.
echo =======================================================
pause
