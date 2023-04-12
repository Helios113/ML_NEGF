
import subprocess

commands = [
    # ["python", "/home/preslav/Projects/ML_NEGF/main.py", "--name", "test3", "--residu", "1", "--tar", "pot"],
    # ["python", "/home/preslav/Projects/ML_NEGF/main.py", "--name", "test5", "--tar", "charge"],
    ["python", "/Users/preslav/Projects/ML_NEGF/main.py", "--name", "test11", "--tar", "charge", "--gpu"]
    # ["python", "/home/preslav/Projects/ML_NEGF/main.py", "--name", "test7", "--residu", "1", "--tar", "charge"]
]

for cmd in commands:
    process = subprocess.Popen(cmd, 
                            stdout=subprocess.PIPE, 
                            universal_newlines=True)