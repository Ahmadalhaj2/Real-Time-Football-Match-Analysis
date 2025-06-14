@echo off
setlocal enabledelayedexpansion

:: Define the directory to process.
:: By default, it processes the current directory where the script is run.
:: If you want to specify a different directory, uncomment the line below and modify it:
set "TARGET_DIR=C:\Users\ADMIN\Desktop\ahmad\data_train\football-players-detection-1\valid\labels"
:: set "TARGET_DIR=."  <-- This line is now commented out

echo.
echo Starting file processing in: %TARGET_DIR%
echo.

:: Loop through all .txt files in the target directory
for %%f in ("%TARGET_DIR%\*.txt") do (
    echo Processing file: "%%f"

    set "TEMP_FILE=%%f.tmp"
    del "!TEMP_FILE!" 2>nul

    :: Read each line from the current file
    for /f "usebackq tokens=*" %%a in ("%%f") do (
        set "line=%%a"
        set "first_char=!line:~0,1!"

        :: Check the first character
        if "!first_char!"=="0" (
            :: If it starts with 0, write the line as is
            echo !line!>>"!TEMP_FILE!"
        ) else if "!first_char!"=="1" (
            :: If it starts with 1, change it to 1 (no change needed for 1)
            echo 1!line:~1!>>"!TEMP_FILE!"
        ) else if "!first_char!"=="2" (
            :: If it starts with 2, change it to 1
            echo 1!line:~1!>>"!TEMP_FILE!"
        ) else if "!first_char!"=="3" (
            :: If it starts with 3, change it to 1
            echo 1!line:~1!>>"!TEMP_FILE!"
        ) else (
            :: For any other starting character, keep the line as is
            :: You might want to adjust this behavior based on your needs
            echo !line!>>"!TEMP_FILE!"
        )
    )

    :: Replace the original file with the temporary one
    if exist "!TEMP_FILE!" (
        del "%%f"
        ren "!TEMP_FILE!" "%%~nxf"
        echo Successfully updated "%%f"
    ) else (
        echo Error: Temporary file "!TEMP_FILE!" was not created for "%%f".
    )
    echo.
)

echo All .txt files processed.
pause
endlocal