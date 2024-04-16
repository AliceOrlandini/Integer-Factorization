@echo off
setlocal enabledelayedexpansion

REM Check the number of parameters
if "%~4"=="" (
    echo Error: Illegal number of parameters, please insert: executable path, number of iterations, max number of threads, and number to be factorized.
    exit /b 1
)

REM Parameters:
REM Path of the executable
set "EXECUTABLE=%~1"
REM Number of iterations
set "NUM_ITERATIONS=%~2"
REM Maximum number of threads to be tested
set "MAX_THREADS=%~3"
REM Number to be factorized
set "NUMBER=%~4"

REM Local variables:
REM Current Time
for /f "tokens=1-6 delims=:-./ " %%a in ('echo %date%_%time%') do (
    set "TIME=%%a-%%b-%%c_%%d-%%e-%%f"
)
REM Output file
set "OUTPUT_FILE=output_%TIME%.csv"

echo Executing the program with the following parameters:
echo.
echo Executable Path: %EXECUTABLE%
echo Number of iterations: %NUM_ITERATIONS%
echo Maximum number of threads: %MAX_THREADS%
echo Number to be factorized: %NUMBER%
echo.

REM Creating the structure of the output file (CSV)
echo num_threads,iteration,execution_time> %OUTPUT_FILE%

REM Execute the program
for /L %%t in (1,1,%MAX_THREADS%) do (
    for /L %%i in (0,1,%NUM_ITERATIONS%) do (
        echo Running iteration %%i with %%t threads
        set /a "IDX=%%t-1"
        echo|set /p="%%t,%%i," >> %OUTPUT_FILE%
        call "%EXECUTABLE%" %%t %NUMBER% 0 >> %OUTPUT_FILE%
    )
)

endlocal
