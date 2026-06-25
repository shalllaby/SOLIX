import os
import time
import subprocess

def restart_server():
    port = 8000
    print(f"Searching for process on port {port}...")
    try:
        # Get netstat output to find PID
        output = subprocess.check_output("netstat -ano", shell=True).decode('utf-8', errors='ignore')
        target_pids = set()
        for line in output.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.strip().split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    target_pids.add(pid)
        
        for pid in target_pids:
            print(f"Terminating stale PID {pid} on port {port}...")
            subprocess.call(f"taskkill /F /PID {pid}", shell=True)
            time.sleep(1)
    except Exception as e:
        print("Error during port cleanup:", e)

    # Make sure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    print("Launching FastAPI backend/main.py in a new console...")
    # Launch uvicorn server in a new console window so it remains running independently
    creation_flags = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    p = subprocess.Popen(
        ["python", "backend/main.py"],
        stdout=open("logs/server_out.log", "w", encoding="utf-8"),
        stderr=open("logs/server_err.log", "w", encoding="utf-8"),
        creationflags=creation_flags
    )
    print(f"Server spawned with process ID {p.pid}")

if __name__ == "__main__":
    restart_server()
