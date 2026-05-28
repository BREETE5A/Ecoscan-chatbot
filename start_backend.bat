@echo off
title EcoScan Backend + ngrok
color 0A

echo ========================================
echo    EcoScan Backend - Demarrage
echo ========================================
echo.

REM === CONFIGURATION (a remplir une seule fois) ===
set NGROK_TOKEN=VOTRE_AUTHTOKEN_ICI
set NGROK_DOMAIN=VOTRE_DOMAINE.ngrok-free.app
REM ================================================

REM Configurer ngrok si pas encore fait
echo [1/3] Configuration ngrok...
ngrok config add-authtoken %NGROK_TOKEN% >nul 2>&1

REM Activer l'environnement virtuel Python si present
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [2/3] Environnement Python active.
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [2/3] Environnement Python active.
) else (
    echo [2/3] Utilisation Python systeme.
)

REM Demarrer FastAPI en arriere-plan
echo [3/3] Demarrage FastAPI sur port 8000...
start "FastAPI Backend" cmd /c "uvicorn app.main:app --host 0.0.0.0 --port 8000 && pause"

REM Attendre que FastAPI soit pret
timeout /t 3 /nobreak >nul

REM Demarrer ngrok avec domaine statique
echo.
echo ========================================
echo  URL publique permanente :
echo  https://%NGROK_DOMAIN%
echo ========================================
echo.
echo  Copiez cette URL dans api_constants.dart
echo  Appuyez sur Ctrl+C pour arreter
echo ========================================
echo.

ngrok http --url=%NGROK_DOMAIN% 8000

pause
