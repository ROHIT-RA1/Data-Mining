
import os


def folderCreation():
  # Creating folder Slices
 os.mkdir(os.path.join('./testPatient','Slices'))

 # Creating folder Boundaries
 os.mkdir(os.path.join('./testPatient','Clusters'))


def SubFolderCreation():
    # Creating Sub-Folders in Slices and Boundaries Folders
    for p in range(1,113):
       if(os.path.exists('./testPatient/Clusters/OUT_{}.format(p)')==False):
        os.mkdir(os.path.join('./testPatient/Clusters','OUT_{}'.format(p)))

       if(os.path.exists('./testPatient/Slices/OUT_{}'.format(p))==False):
        os.mkdir(os.path.join('./testPatient/Slices','OUT_{}'.format(p)))
    

def countReqFiles(dir_path):
    count = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:    
            if file.endswith('thresh.png'):
                count += 1
    return count