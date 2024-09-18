@echo off

for %%f in (*.mp4) do (
    echo process: "%%f"
    D:\Programs\OpenFace_2.2.0_win_x64\FeatureExtraction.exe -f "%%f"  -3Dfp -pose  -aus -gaze -hogalign
)
echo process done
pause
