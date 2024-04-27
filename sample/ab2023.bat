@echo off

setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x86_amd64

call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018.3.210\windows\bin\ifortvars.bat" intel64

call "C:\SIMULIA\Commands\abq2023hf1.bat" %*

endlocal

