from zeroconf import ServiceBrowser, ServiceListener, Zeroconf

anchor_service_name = 'cranebot-server-anchor'

class CranebotListener(ServiceListener):
    def __init__(self):
        super().__init__()
        self.anchors = []
        self.grippers = []

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print(f"Service {name} updated")

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print(f"Service {name} removed")

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        print(f"Service {name} added, service info: {info}")
        if name.split(".")[0] == anchor_service_name:
            self.anchors.append((name, info))


zeroconf = Zeroconf()
listener = CranebotListener()
browser = ServiceBrowser(zeroconf, "_http._tcp.local.", listener)
try:
    input("Press enter to exit...\n\n")
finally:
    zeroconf.close()