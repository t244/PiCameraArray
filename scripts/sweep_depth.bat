@echo off
REM ============================================================================
REM  Depth Sweep - d=700 to d=800, step=2mm, all datasets
REM
REM  Usage:
REM    scripts\sweep_depth.bat              (frame=0)
REM    scripts\sweep_depth.bat 5            (frame=5)
REM ============================================================================

setlocal enabledelayedexpansion

REM --- Configurable parameters ------------------------------------------------
set FRAME_NUM=%~1
if "%FRAME_NUM%"=="" set FRAME_NUM=0

REM Zero-pad frame number to 6 digits
set "PADDED=000000%FRAME_NUM%"
set "FRAME=!PADDED:~-6!"

set CALIB=calibration_results.npz
set REF=5
set D_START=700
set D_END=800
set D_STEP=2
REM ----------------------------------------------------------------------------

echo Frame: %FRAME% (input: %FRAME_NUM%)
echo Depth: %D_START% to %D_END% mm (step %D_STEP%)
echo.

for /D %%S in (packed_data\*) do (
    if exist "%%S\%FRAME%" (
        for %%N in (%%S) do set DSNAME=%%~nxN
        set INDIR=%%S\%FRAME%
        set SWEEPDIR=outputs\!DSNAME!\%FRAME%\depth_sweep

        echo === !DSNAME! / %FRAME% ===
        if not exist "!SWEEPDIR!" mkdir "!SWEEPDIR!"

        for /L %%D in (%D_START%,%D_STEP%,%D_END%) do (
            echo   d=%%D mm ...
            python analyze\mpi_sar.py !INDIR! ^
                --calib %CALIB% --method defencing_v2 --ref-idx %REF% ^
                --focus-depth %%D ^
                -o !SWEEPDIR!\tmp_%%D

            if exist "!SWEEPDIR!\tmp_%%D\result_defencing_v2.png" (
                copy /Y "!SWEEPDIR!\tmp_%%D\result_defencing_v2.png" "!SWEEPDIR!\%%D.png" >nul
            )
            if exist "!SWEEPDIR!\tmp_%%D" rmdir /S /Q "!SWEEPDIR!\tmp_%%D"
        )
        echo.
    )
)

echo All done. Results in outputs\*\%FRAME%\depth_sweep\
pause
