import logging
import subprocess

def forget_all_wifi_networks():
    """
    Finds and deletes all saved WiFi networks using nmcli.
    """
    logging.info("Gathering saved WiFi networks to forget...")
    
    # Get a list of all connections in terse format (UUID:TYPE)
    result = subprocess.run(
        ['nmcli', '--terse', '--fields', 'UUID,TYPE', 'connection', 'show'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logging.error(f"Failed to fetch connections: {result.stderr.strip()}")
        return

    # Parse the output and delete any connection that is a wireless network
    deleted_count = 0
    for line in result.stdout.splitlines():
        if not line:
            continue
            
        parts = line.split(':')
        if len(parts) >= 2:
            uuid = parts[0]
            conn_type = parts[1]
            
            # '802-11-wireless' is the NetworkManager type for WiFi connections
            if conn_type == '802-11-wireless':
                logging.info(f"Deleting WiFi network profile (UUID: {uuid})...")
                del_result = subprocess.run(
                    ['nmcli', 'connection', 'delete', 'uuid', uuid], 
                    capture_output=True, 
                    text=True
                )
                if del_result.returncode == 0:
                    deleted_count += 1
                else:
                    logging.warning(f"Failed to delete {uuid}: {del_result.stderr.strip()}")

    logging.info(f"Finished. Removed {deleted_count} WiFi network(s).")