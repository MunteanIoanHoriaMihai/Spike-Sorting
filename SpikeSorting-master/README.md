# Spike Sorting Project
# 2021/2022: 
PhD of Ardelean Eugen-Richard  
Dissertation Thesis (Second Year) of Lazea Diana-Georgeta and Hristache Alexandru-Andrei  
Dissertation Thesis (First Year) of Aldea Roxana-Ioana and Muntean Ioan Horia Mihai  
License Thesis of Muresan Denisa-Bianca, Alungei Paul-Catalin, Ciure Raluca-Dana  
Publication: "Weighted PCA based on statistical properties of features for Spike Sorting"  

# 2020/2021: 
Dissertation Thesis (Second Year) of Ardelean Eugen-Richard and Coporiie Andreea  
Dissertation Thesis (First Year) of Lazea Diana-Georgeta and Hristache Alexandru-Andrei, collaborator: Andreea Gui  
License Thesis of Aldea Roxana-Ioana and Negru Vlad  
  
# 2019/2020:
Dissertation Thesis (First Year) of Ardelean Eugen-Richard and Coporiie Andreea  
License Thesis of Gui Andreea, Lazea Diana-Georgeta and Hristache Alexandru-Andrei  

# 2018/2019:
License Thesis of Ardelean Eugen-Richard and Stanciu Alexander  
Publication: "Space Breakdown Method" - DOI: 10.1109/ICCP48234.2019.8959795

# Guidance
Coordinators: Dinsoreanu Mihaela, Potolea Rodica, Lemnaru Camelia (UTCN)  
Collaborators: Muresan Raul Cristian, Moca Vlad Vasile (TINS)  

# Datasets: 
https://www135.lamp.le.ac.uk/hgr3/

# Project Structure: 
- Each has a "main_name.py" file for running their own tests <br />
- datasets (folder): location in which datasets will be stored (it is not uploaded to git due to large file sizes)
- feature_extraction (folder): code for feature extraction methods implemented
- figures (folder): save location for all plots (to be saved in subfolders) - not all plot subfolders are to be uploaded to git
- libraries (folder): folder to save libraries to be used that cannot be installed
- results (folder): spreadsheet type files for evaluation purposes
- utils (folder): contains general functions
  - benchmark (folder): code for benchmarking different algorithms
  - dataset_parsing (folder): code for parsing different datasets used
  - documents (folder): location of files in doc format
  - sbm (folder): code for SBM algorithm
  - constants.py : file to store different constants for the code
  - scatter_plot.py : file to store plotting methods
- requirements.txt - general library management file, when a new library is added, this file will be updated with the library name and its version <br />