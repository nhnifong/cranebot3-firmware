# main entry point for pyinstaller (windows)

from nf_robot.host.observer import main

import os
import sys

# Add 'src' to path so nf_robot is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

def get_windows_config_path():
    """Returns a path to %LOCALAPPDATA%/nf_robot/configuration.json"""
    # LOCALAPPDATA is usually C:\Users\<User>\AppData\Local
    base_dir = os.environ.get('LOCALAPPDATA', os.path.expanduser("~"))
    app_dir = os.path.join(base_dir, "nf_robot")
    
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)
        
    return os.path.join(app_dir, "configuration.json")

if __name__ == "__main__":
    from nf_robot.host.observer import main
    
    # Inject the --config argument if it's not already provided
    # This ensures the EXE uses LocalAppData by default
    if "--config" not in sys.argv:
        config_path = get_windows_config_path()
        sys.argv.extend(["--config", config_path])
        print(f"Using config at: {config_path}")

    # Launch the controller
    main()