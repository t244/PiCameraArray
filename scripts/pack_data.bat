@echo off
REM ============================================================================
REM  Pack Data - 20260217 dataset
REM
REM  Source:  20260217_dataset\e00\<timestamp>\*.png ...
REM  Output:  packed_data\<timestamp>\<counter>\e00.png ... e15.png
REM
REM  Usage:
REM    scripts\pack_data.bat                 (default: offset=2, 3rd from latest)
REM    scripts\pack_data.bat --offset 0      (latest timestamp)
REM    scripts\pack_data.bat --dry-run       (preview without copying)
REM ============================================================================

setlocal

set DATASET=20260217_dataset
set OUTPUT=packed_data

REM Pass through any command-line arguments (e.g. --offset 0 --dry-run)
echo Packing data from %DATASET% ...
echo.

python construct\pack_data_0217.py --dataset %DATASET% --out %OUTPUT% %* --offset 8

echo.
echo Done. Packed data in %OUTPUT%\
pause
