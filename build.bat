@echo off
setlocal

REM Always run from the folder containing this batch file.
cd /d "%~dp0"

if not exist results mkdir results

echo Compiling sequential program...

cl /nologo /EHsc /O2 /W4 /std:c++17 ^
sequential_analytics.cpp ^
/Fe:sequential_analytics.exe

if errorlevel 1 (
    echo.
    echo Sequential compilation failed.
    pause
    exit /b 1
)

echo.
echo Compiling MPI program...

cl /nologo /EHsc /O2 /W4 /std:c++17 ^
/I"C:\MPI\MS_MPI\SDK\Include" ^
mpi_analytics.cpp ^
/Fe:mpi_analytics.exe ^
/link ^
/LIBPATH:"C:\MPI\MS_MPI\SDK\Lib\x64" ^
msmpi.lib

if errorlevel 1 (
    echo.
    echo MPI compilation failed.
    pause
    exit /b 1
)

echo.
echo Both programs compiled successfully.
pause

endlocal

