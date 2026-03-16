import sys
import subprocess
import shlex

def run_from_stdin():
    # Read from the pipe/redirected file
    for line in sys.stdin:
        # Clean up whitespace and skip empty lines
        command_str = line.strip()
        if not command_str or command_str.startswith("#"):
            continue
            
        print(f"🚀 Executing command: {command_str}")
        
        # shlex.split transforms 'python script.py "my file.txt"' 
        # into ['python', 'script.py', 'my file.txt']
        command_list = shlex.split(command_str)
        
        try:
            # We run the command and wait. Errors in the subprocess 
            # won't stop this loop from moving to the next line.
            result = subprocess.run(command_list, capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"✅ Success.\n")
            else:
                print(f"⚠️ Failed with exit code {result.returncode}. Moving to next...\n")
                
        except Exception as e:
            print(f"❌ System Error executing command: {e}\n")

if __name__ == "__main__":
    run_from_stdin()
