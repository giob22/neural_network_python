import glob

for fpath in glob.glob("tests_script/test_*.py"):
    with open(fpath, "r") as f:
        lines = f.readlines()
    
    # Find where the execution starts (e.g., 'csv_rows  = []' or 'print('Esecuzione test Leakage B...')')
    exec_start = 0
    for i, line in enumerate(lines):
        if line.startswith("csv_rows  = []") or line.startswith("print(\"Esecuzione test Leakage B...\")"):
            exec_start = i
            break
            
    if exec_start == 0:
        continue
        
    new_lines = lines[:exec_start] + ["if __name__ == '__main__':\n"] + ["    " + line for line in lines[exec_start:]]
    
    with open(fpath, "w") as f:
        f.writelines(new_lines)

