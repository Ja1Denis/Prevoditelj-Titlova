@echo off
setlocal

pyinstaller ^
    --onefile ^
    --windowed ^
    --name="Sinkronizator Titlova" ^
    improved_translator.py

endlocal
