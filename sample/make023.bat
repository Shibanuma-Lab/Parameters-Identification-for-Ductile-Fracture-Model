cd /d %~dp0
set VAR=%CD%
for /f %%A in ("%VAR%") do set CURRENT_DIR_NAME=%%~nxA
type nul > ../%CURRENT_DIR_NAME%.023
type nul > ../%CURRENT_DIR_NAME%.log
ab2023 job=subroutine_practice user=UMAT2(2020-03-31)-6 cpus=1