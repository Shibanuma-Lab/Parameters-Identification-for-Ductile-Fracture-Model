# Parameters-Identification-for-Ductile-Fracture-Model
This project provides Python code for identifying ductile parameters (the hardening exponent n, the fracture strain εf0, material parameters A and B) at once.  <br>
<br>

# Overview
This project provides Python code for identifying ductile parameters (the hardening exponent n, the fracture strain εf0, material dependant parameters A and B) based on Tensile and Arcan experiment data, along with an example.  
<br>

# Description
This project aims to provide a tool for conveniently identifying ductile parameters all at once. It streamlines both the process of importing experimental data and the algorithmic analysis, facilitating parameter identification. Key features include:

#### 1. Single-step Completion: 
Supports the sequential identification of ductile parameters (n, εf0, B, and A), allowing the entire identification process to be run at once without the need for step-by-step operations.

#### 2. Parallel Task Mode: 
Offers a parallel task mode, which can significantly reduce the identification process time by approximately 30-40%.

#### 3. Stepwise Parameter Identification: 
The identification process for each parameter can be calculated separately, and the identification results for each parameter are provided in CSV files.

#### 4. Flexible Interruption and Resumption: 
During the parameter identification process, interruptions are permitted at any time (assuming submitted Abaqus jobs have been completed). When rerunning the program, it can automatically resume parameter identification from the interruption phase.

#### 5. Multiple Configuration Options: 
Provides multiple customization options to meet diverse identification needs.

#### 6. Automatic Result Organization: 
After completing identifications, the results are automatically organized for convenient viewing and analysis.
<br>
<br>


# Execution method
## 1. Install Required Libraries: 
Ensure that all required libraries are installed.
## 2. Ensure Files Are in the Same Folder: 
Ensure that all code files and data files are located in the same folder. The input files include:
##### INP files for both Tensile and Arcan tests
##### Data obtained from DIC (named as "Arcan test INP file name"+"_Arcan"/"_Load", etc.)
##### Curves obtained from Tensile experiments (named as "Material name"+"_ss")
##### .bat file and UMAT(.f)
## 3. Configure Options: 
Before running the Python file, some customization options can be modified. These options are available in the "ALLEXE.py". Adjust these options according to specific requirements, such as setting jobnumber=3 for parallel mode or jobnumber=1 for normal mode, computing one job at a time.
## 4. Attention: 
Please note that "make023.bat" is the file used for batch submission in Abaqus. Please modify it according to the computer you are running this code and the version of Abaqus.
## 5. Run the Python File: 
Execute the Python file "ALLEXE.py".  
<br>
<br>

# Example Application

The folder named "example" contains an example. It contains all input files necessary for execution. After identification, a new folder named "ALLResults" will be generated, where the parameter identification results and important process files will be stored.

<br>


# References
[1] Tu S, Suzuki S, Yu Z, et al. Hybrid experimental-numerical strategy for efficiently and accurately identifying post-necking hardening and ductility diagram parameters[J]. International Journal of Mechanical Sciences, 2022, 219: 107074.



<br>















