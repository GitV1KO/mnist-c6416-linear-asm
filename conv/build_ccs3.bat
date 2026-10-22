@echo off
setlocal
rem Usage: build_ccs3.bat "C:\CCStudio_v3.1\C6000\cgtools"
set "CNN_CGT_ROOT=%~1"
if not defined CNN_CGT_ROOT set "CNN_CGT_ROOT=C:\CCStudio_v3.1\C6000\cgtools"
if not exist "%CNN_CGT_ROOT%\bin\cl6x.exe" (
    echo cl6x.exe not found. Pass the course C6000 cgtools directory.
    exit /b 1
)
cd /d "%~dp0"
if not exist build mkdir build
"%CNN_CGT_ROOT%\bin\cl6x.exe" -mv6400 -o3 -k -i. -i"%CNN_CGT_ROOT%\include" -frbuild -fsbuild -c main.c cnn.c cnn_dotprod128_c.c clock_demo.c cnn_dotprod128.sa
if errorlevel 1 exit /b 1
"%CNN_CGT_ROOT%\bin\cl6x.exe" -mv6400 -z -c -heap0x800 -stack0x2000 -m"build\cnn_linear.map" -o"build\cnn_linear.out" -i"%CNN_CGT_ROOT%\lib" "build\main.obj" "build\cnn.obj" "build\cnn_dotprod128_c.obj" "build\clock_demo.obj" "build\cnn_dotprod128.obj" C6416.cmd -l"%CNN_CGT_ROOT%\lib\rts6400.lib"
if errorlevel 1 exit /b 1
echo Built build\cnn_linear.out. Load it in CCS using the course board configuration.
endlocal
