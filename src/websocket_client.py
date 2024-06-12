import time
import websockets
from typing import Union
from datetime import datetime
from syft.generic.pointers.object_wrapper import ObjectWrapper
from syft.messaging.message import ObjectMessage
from syft.generic.tensor import AbstractTensor
from syft.generic.frameworks.types import FrameworkTensorType
import binascii
import syft as sy

from syft.workers.websocket_client import WebsocketClientWorker
from syft.messaging.message import ObjectRequestMessage


class MyWebsocketClientWorker(WebsocketClientWorker):

    async def async_command(self, command: dict, return_id=None):
        try:
            self.close()

            async with websockets.connect(
                    self.url, timeout=60, max_size=None, ping_timeout=60
            ) as websocket:

                message = self.create_worker_command_message(**command)

                serialized_message = sy.serde.serialize(message)
                await websocket.send(str(binascii.hexlify(serialized_message)))
                await websocket.recv()

            self.connect()

            if return_id is None:
                return
            else:
                msg = ObjectRequestMessage(return_id, None, "")
                serialized_message = sy.serde.serialize(msg)
                response = self._send_msg(serialized_message)
                return response

        except Exception as e:
            print(f"An error occurred during async_command: {str(e)}")
            return None
