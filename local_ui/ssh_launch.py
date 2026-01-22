import platform
import subprocess
import shutil
import sys
import os

def launch_ssh_terminal(address, user="pi"):
    """
    Opens a new terminal window and runs the SSH command for the specific platform.
    """
    current_os = platform.system()
    ssh_command = f"ssh {user}@{address}"
    
    print(f"Detected OS: {current_os}")
    print(f"Attempting to launch: {ssh_command}")

    if current_os == "Windows":
        # Windows handling
        # 'start' is a shell command in Windows, so shell=True is needed.
        # 'cmd /k' keeps the window open after the command finishes (or fails)
        # so you can see the output.
        subprocess.run(f'start cmd /k "{ssh_command}"', shell=True)

    elif current_os == "Darwin":
        # macOS handling
        # Uses AppleScript via 'osascript' to tell the Terminal application to run the script.
        # This opens a new window/tab in the default Terminal app.
        apple_script_cmd = f'tell application "Terminal" to do script "{ssh_command}"'
        subprocess.run(["osascript", "-e", apple_script_cmd])
        
        # Optional: Bring Terminal to front
        subprocess.run(["osascript", "-e", 'tell application "Terminal" to activate'])

    elif current_os == "Linux":
        # Linux handling
        # Linux is fragmented with many terminal emulators. 
        # We check for common ones in order of preference.
        
        terminals = [
            # format: (executable, argument_to_execute_command)
            ("gnome-terminal", "--"),  # Gnome
            ("konsole", "-e"),          # KDE
            ("xfce4-terminal", "-x"),   # XFCE
            ("xterm", "-e"),            # Standard X11
            ("terminator", "-x")        # Terminator
        ]
        
        found_terminal = False
        
        for term, flag in terminals:
            if shutil.which(term):
                print(f"Found terminal emulator: {term}")
                # We use string splitting for the command to ensure arguments are passed correctly
                cmd_args = [term, flag, "bash", "-c", f"{ssh_command}; exec bash"]
                
                # Gnome-terminal requires a slightly different syntax for the command arguments
                if term == "gnome-terminal":
                    cmd_args = [term, "--", "bash", "-c", f"{ssh_command}; exec bash"]
                    
                # 'exec bash' at the end keeps the terminal open after SSH disconnects
                subprocess.Popen(cmd_args)
                found_terminal = True
                break
        
        if not found_terminal:
            print("Error: Could not find a supported terminal emulator (gnome-terminal, konsole, xterm, etc).")
            print(f"You can run the command manually: {ssh_command}")

    else:
        print(f"Sorry, automatic terminal launching is not supported for {current_os}.")