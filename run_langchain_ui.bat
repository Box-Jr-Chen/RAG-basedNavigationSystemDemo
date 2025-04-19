@echo off
echo  啟動 Conda 環境中...


REM 啟動你指定的環境
CALL conda activate langchain_env

echo  啟動 LangChain API（querying.py）...
start cmd /k uvicorn querying:app --reload

timeout /t 3 >nul

echo 啟動 Tkinter 介面（tkinter_sound.py）...
python tkinter_sound.py

pause