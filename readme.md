##  run python file (.py) on hadoop distributed file system 

 1. yarn_run.sh:  
    use yarn_run.sh starting procedure  
    data files can be placed on hdfs  
    
 2. run.sh  
    load miniconda3 package  
    load data  
    put the results to hdfs  
 3. how to load python module  
     import sys,os  
     sys.path.append(os.getcwd())  
     from meta_data import xxx  


