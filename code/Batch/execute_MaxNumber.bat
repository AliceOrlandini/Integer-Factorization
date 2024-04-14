@echo off
setlocal enabledelayedexpansion

REM Check the number of parameters
if "%~5"=="" (
    echo Error: Illegal number of parameters, please insert: executable path, number of iterations, number of threads, max number to be factorized, and step of the number.
    exit /b 1
)

REM Parameters:
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
    for /L %%i in (0,1,%NUM_ITERATIONS%) do (
        echo Running iteration %%i with the number %%n
        echo|set /p="%%n,%%i," >> %OUTPUT_FILE%
        call "%EXECUTABLE%" %NUM_THREADS% %%n 0 >> %OUTPUT_FILE%
    )
)

endlocal
