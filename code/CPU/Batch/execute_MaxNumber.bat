@echo off
setlocal enabledelayedexpansion

REM This script is used to execute an executable for NUM_ITERATIONS times,
REM with NUM_THREADS threads, and changing the number to be factorized from STEP to MAX_NUMBER,
REM with the step of STEP. The output is saved in a CSV file.

REM Check the number of parameters
if "%~5"=="" (
    echo Error: Illegal number of parameters, please provide:
    echo 1. Executable path
    echo 2. Number of iterations
    echo 3. Number of threads
    echo 4. Max number to be factorized
    echo 5. Step of the number
    exit /b 1
)

REM PARAMETERS

REM Path of the executable
set "EXECUTABLE=%~1"
REM Number of iterations
set "NUM_ITERATIONS=%~2"
REM Number of threads used in the execution
set "NUM_THREADS=%~3"
REM Max Number to be factorized
set "MAX_NUMBER=%~4"
REM Step used for incrementing the number to be factorized
set "STEP=%~5"

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
echo Number of threads: %NUM_THREADS%
echo Max number to be factorized: %MAX_NUMBER%
echo Step of the number: %STEP%
echo.

REM Creating the structure of the output file (CSV)
echo number,iteration,execution_time> %OUTPUT_FILE%

REM Execute the program
for /L %%n in (%STEP%,%STEP%,%MAX_NUMBER%) do (
    for /L %%i in (1,1,%NUM_ITERATIONS%) do (
        echo Running iteration %%i with the number %%n
        echo|set /p="%%n,%%i," >> %OUTPUT_FILE%
        call "%EXECUTABLE%" %NUM_THREADS% %%n 0 >> %OUTPUT_FILE%
    )
)

endlocal
