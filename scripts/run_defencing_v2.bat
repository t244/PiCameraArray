@echo off
REM ============================================================================
REM  Defencing v2 - Run on all datasets in packed_data
REM
REM  Usage:
REM    scripts\run_defencing_v2.bat              (frame=0, d=750)
REM    scripts\run_defencing_v2.bat 5            (frame=5, d=750)
REM    scripts\run_defencing_v2.bat 5 700        (frame=5, d=700)
REM ============================================================================

setlocal enabledelayedexpansion

REM --- Configurable parameters ------------------------------------------------
set FRAME_NUM=%~1
if "%FRAME_NUM%"=="" set FRAME_NUM=0

set DEPTH=%~2
if "%DEPTH%"=="" set DEPTH=750

REM Zero-pad frame number to 6 digits
set "PADDED=000000%FRAME_NUM%"
set "FRAME=!PADDED:~-6!"

set CALIB=calibration_results.npz
set REF=5
set COMMON=--calib %CALIB% --method defencing_v2 --ref-idx %REF% --save-masks --focus-depth %DEPTH%
REM ----------------------------------------------------------------------------

echo Frame: %FRAME% (input: %FRAME_NUM%)
echo Depth: %DEPTH% mm
echo.

set COUNT=0
set TOTAL=0

REM Count datasets
for /D %%S in (packed_data\*) do (
    if exist "%%S\%FRAME%" set /A TOTAL+=1
)

REM Process each dataset
for /D %%S in (packed_data\*) do (
    if exist "%%S\%FRAME%" (
        set /A COUNT+=1

        REM Extract dataset name (e.g. 20260217_084516)
        for %%N in (%%S) do set DSNAME=%%~nxN

        set INDIR=%%S\%FRAME%
        set OUTDIR=outputs\!DSNAME!\%FRAME%\v2\d%DEPTH%

        echo [!COUNT!/!TOTAL!] !DSNAME! / %FRAME% ...
        python analyze\mpi_sar.py !INDIR! %COMMON% -o !OUTDIR!
        echo.
    )
)

echo All done. Processed %COUNT% datasets.
echo Results in outputs\*\%FRAME%\v2\d%DEPTH%\
pause
