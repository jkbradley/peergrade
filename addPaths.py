# Add all code paths.
import os, sys

def addPaths(initdir):
    for root, dirs, files in os.walk(initdir):
        sys.path.append(os.path.abspath(initdir))
        for name in dirs:
            sys.path.append(os.path.abspath(os.path.join('..',name)))
