import subprocess
import sys

def main(folder_name):
    subprocess.run(["python", "gunthy_ai.py", folder_name])

if __name__ == "__main__":
    folder_name = sys.argv[1] if len(sys.argv) > 1 else 'gunthytrainingdata'
    main(folder_name)