"""Bounded retries for explicitly transient, idempotent controller transport operations."""
import time
import modal
import aiohttp
from grpclib.exceptions import GRPCError, StreamTerminatedError
from grpclib.const import Status
from stage1.budget import BudgetStop


def transient(error):
    if isinstance(error, (FileNotFoundError, PermissionError)):
        return False
    if isinstance(error, GRPCError):
        return error.status in (Status.UNAVAILABLE, Status.DEADLINE_EXCEEDED)
    return isinstance(error, (ConnectionError, TimeoutError, modal.exception.ConnectionError,
                              aiohttp.ClientConnectionError, StreamTerminatedError))


def retry_io(operation, *, deadline=None, attempts=6, sleep=time.sleep, clock=time.time):
    for index in range(attempts):
        if deadline is not None and clock() >= deadline:
            raise BudgetStop('Existing call deadline reached during transport recovery')
        try:
            return operation()
        except Exception as error:
            if not transient(error) or index+1==attempts:
                raise
        delay=min(2**index,30)
        if deadline is not None:
            delay=min(delay,max(0,deadline-clock()))
        sleep(delay)
