cd /d %~dp0
set VAR=%CD%
for /f %%A in ("%VAR%") do set CURRENT_DIR_NAME=%%~nxA
type nul > ../%CURRENT_DIR_NAME%.023
type nul > ../%CURRENT_DIR_NAME%.log
ab2023 terminate job=T02_n_0.072136