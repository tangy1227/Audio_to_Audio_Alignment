## Audio to Audio Alignment Tool
A tool that automatically analyzes the similarity between two different audio, and aligns them together using a dynamic time warping algorithm. The tool can take in multiple audio snippet recordings of a music piece and align them to the main track. The result will be output as a REAPER project with the main track and aligned audio snippet tracks. 

## Installation
Download all the code here to run the tool. The `requirements.txt` has all the required python packages to run the code.

## Usage
- Locate `run.py` and run the python file. After running the file, a `File Directory Input Helper` GUI will display
![alt text](https://github.com/tangy1227/Audio_to_Audio_Alignment/blob/main/images/Screen%20Shot%202022-04-27%20at%2023.01.09.jpg)
- Follow instructions on the GUI by each row and select the directory. The red text will show the status
- The tool will automatically generator a REAPER project. Click “Open Project” to open the REAPER file, or find the REAPER file on your chosen output location.
