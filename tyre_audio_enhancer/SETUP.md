# Tyre Audio Enhancer Setup

## Step 1 — Install VB-Cable (free, one click)
Download from: https://vb-audio.com/Cable/
Run the installer as Administrator.
Restart if prompted.

## Step 2 — Route iRacing through VB-Cable
Right-click speaker icon in Windows taskbar
→ Sound Settings → Output
→ Select "CABLE Input (VB-Audio Virtual Cable)"

iRacing's audio now flows through VB-Cable into Python.

## Step 3 — Install Python dependencies
pip install sounddevice numpy scipy

## Step 4 — Find your device names
python main.py --list-devices

Look for "CABLE Output" (this is your input — what comes OUT of VB-Cable)
Look for your AX50 headphones name

## Step 5 — Run calibration first
python main.py --calibrate --input "CABLE Output"
Rev the engine during calibration to detect harmonics.

## Step 6 — Run the enhancer
python main.py --input "CABLE Output" --output "Headphones (AX50)"

## Tuning
Too much engine still audible:  increase --engine-cut (e.g. -30)
Tyre sounds too quiet:          increase --tyre-boost (e.g. +15)
Tyre sounds too loud/distorted: decrease --tyre-boost (e.g. +8)
