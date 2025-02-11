import zeroconf

def register_mdns_service(name, service_type, port, properties={}):
    """Registers an mDNS service on the network."""

    zc = zeroconf.Zeroconf()
    info = zeroconf.ServiceInfo(
        service_type,
        name + "." + service_type,
        port=port,
        properties=properties,
    )

    zc.register_service(info)
    print(f"Registered service: {name} ({service_type}) on port {port}")

    try:
        while True:
            pass  # Keep the service running
    except KeyboardInterrupt:
        pass
    finally:
        zc.unregister_service(info)
        print("Service unregistered")

if __name__ == "__main__":
    register_mdns_service("my_service", "_http._tcp.local.", 8080)