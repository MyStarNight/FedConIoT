import websockets
import syft as sy
import binascii

from typing import Union
from datetime import datetime
from syft import TrainConfig
from syft.generic.pointers.object_wrapper import ObjectWrapper
from syft.messaging.message import ObjectMessage
from syft.generic.tensor import AbstractTensor
from syft.generic.frameworks.types import FrameworkTensorType
from syft.workers.websocket_client import WebsocketClientWorker


class MyTrainConfig(TrainConfig):
    async def async_wrap_and_send(
            self,
            obj: Union[FrameworkTensorType, AbstractTensor],
            location: WebsocketClientWorker
    ):
        try:
            location.close()

            async with websockets.connect(
                    location.url, timeout=60, max_size=None, ping_timeout=60
            ) as websocket:
                obj_id = sy.ID_PROVIDER.pop()
                print(location.id, obj_id)
                obj_with_id = ObjectWrapper(id=obj_id, obj=obj)
                obj_message = ObjectMessage(obj_with_id)
                bin_message = sy.serde.serialize(obj_message, worker=self.owner)
                await websocket.send(str(binascii.hexlify(bin_message)))
                await websocket.recv()

            location.connect()

            return

        except Exception as e:
            print(f"An error occurred during async_wrap_and_send: {str(e)}")
            return None

    def wrap_and_send(self, obj, location):
        obj_with_id = ObjectWrapper(id=sy.ID_PROVIDER.pop(), obj=obj)
        obj_ptr = self.owner.send(obj_with_id, location)
        obj_id = obj_ptr.id_at_location
        return obj_ptr, obj_id


class PullAndPush:
    def __init__(self):
        self.owner = sy.hook.local_worker

    def send(self, obj, location, ID=None):
        if ID is None:
            ID = sy.ID_PROVIDER.pop()
        obj_with_id = ObjectWrapper(id=ID, obj=obj)
        obj_ptr = self.owner.send(obj_with_id, location)
        obj_id = obj_ptr.id_at_location
        return obj_ptr, obj_id
