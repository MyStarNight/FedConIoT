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


def device_id_to_model_id(ID: str):
    appendix = None
    if len(ID) == 1:
        appendix = '0' + str(ord(ID) - ord('A') + 1)
    elif len(ID) == 2 and ID[0] == ID[1]:
        appendix = '1' + str(ord(ID[0]) - ord('A') + 1)

    return int(str('1'*9)+appendix)


def model_id_to_device_id(model_id: int):
    info = int(str(model_id)[-2:])
    if info > 10:
        info = info - 10 - 1
        model_id = chr(65+info)*2
    elif info <= 10:
        info = info - 1
        model_id = chr(65+info)

    return model_id


def generate_command_dict(command_name, **kwargs):
    kwargs['command_name'] = command_name
    return kwargs
