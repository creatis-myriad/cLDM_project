import glob
import os



def GetPaths(input_dir, name_files=["dcm.pkl", "roi.pkl"], match_patern=None) :
    normpath = os.path.normpath("/".join([input_dir, '**', '*']))
    paths = {}
    for elem in name_files :
        paths[elem] = []

    for file in glob.iglob(normpath, recursive=True):
        if os.path.isfile(file) and True in [name in file for name in name_files]:
            if match_patern != None and match_patern not in file : continue
            for elem in name_files :    
                if elem in file :
                    paths[elem].append(file)
    return paths