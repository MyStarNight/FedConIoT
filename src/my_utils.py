import socket

client_device_mapping_id = {
    "192.168.3.5": "AA",
    "192.168.3.6": "BB",
    "192.168.3.9": "CC",
    "192.168.3.15": "DD",
    "192.168.3.16": "EE",
    # "192.168.3.2": "A",
    # "192.168.3.3": "B",
    # "192.168.3.4": "C",
    # "192.168.3.7": "D",
    # "192.168.3.8": "E",
    # "192.168.3.10": "F",
    # "192.168.3.11": "G",
    # "192.168.3.12": "H",
    # "192.168.3.13": "I",
    # "192.168.3.20": "J",
    # "192.168.3.17": "testing"
}

id_mapping_client_device = {value: key for key, value in client_device_mapping_id.items()}

tree = {
    1: [("AA", "BB"), ("AA", "CC")],
    2: [("BB", "DD"), ("CC", "EE")]
}


def get_host_ip():
    """
    查询本机ip地址
    :return: ip
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def generate_kwarg(ID):
    ip = id_mapping_client_device[ID]
    kwarg = {"host": ip, "port": 9292, "id": ID}
    return kwarg
