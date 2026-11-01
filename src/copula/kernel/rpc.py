"""Newline-delimited JSON over a Unix socket: the wire of the cache manager and the control plane.

A request is {"id", "op", ...}; the response is {"id", "ok": true, "result"} or
{"id", "ok": false, "error": {"kind", "detail"}}. Requests on one connection run concurrently. When a
connection drops its pending requests are cancelled; whatever a handler shields (the manager's
flights, the control plane's jobs) goes on. Sockets are created mode 0600.
"""
import asyncio
import itertools
import json
import os

from .core.canon import canonical
from .core.design import DesignError
from .core.key import KeySpecError
from .core.request import RequestSpecError
from .errors import EvalrtError

LINE_LIMIT = 64 * 1024 * 1024       # prompts and records are large; asyncio's default is 64 KiB
_connections = itertools.count(1)


def error_payload(exc):
    if isinstance(exc, EvalrtError):
        return {'kind': exc.kind, 'detail': exc.detail}
    if isinstance(exc, (RequestSpecError, KeySpecError, DesignError, KeyError, TypeError,
                        ValueError)):
        return {'kind': 'bad_request', 'detail': '%s: %s' % (type(exc).__name__, exc)}
    return {'kind': 'internal', 'detail': repr(exc)}


async def _handle(handler, on_close, reader, writer):
    pending = set()
    lock = asyncio.Lock()
    connection = next(_connections)

    async def send(message):
        data = canonical(message) + b'\n'
        async with lock:
            if writer.is_closing():
                return                              # the requester is gone
            try:
                writer.write(data)
                await writer.drain()
            except (ConnectionError, RuntimeError):
                pass

    async def run(req):
        rid = req.get('id') if isinstance(req, dict) else None
        try:
            if not isinstance(req, dict) or 'op' not in req:
                raise EvalrtError('bad_request', 'a request is an object with an op')
            result = await handler(req, connection)
            await send({'id': rid, 'ok': True, 'result': result})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await send({'id': rid, 'ok': False, 'error': error_payload(exc)})

    try:
        while True:
            try:
                line = await reader.readline()
            except (ConnectionError, ValueError):
                break
            if not line:
                break
            try:
                req = json.loads(line)
            except ValueError:
                await send({'id': None, 'ok': False,
                            'error': {'kind': 'bad_request', 'detail': 'not JSON'}})
                continue
            task = asyncio.ensure_future(run(req))
            pending.add(task)
            task.add_done_callback(pending.discard)
    finally:
        for task in list(pending):
            task.cancel()
        if on_close is not None:
            on_close(connection)
        try:
            writer.close()
        except Exception:
            pass


def claim_socket(path):
    """Remove a stale socket left by a crash; refuse if a live server is listening on it."""
    import socket
    if not os.path.exists(path):
        return
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(path)
    except (ConnectionRefusedError, FileNotFoundError):
        os.unlink(path)
        return
    finally:
        probe.close()
    raise SystemExit('another server is listening on %s' % path)


async def serve_json(handler, path, on_close=None):
    """Serve `handler(req, connection)` on the Unix socket `path` (which must not exist)."""
    server = await asyncio.start_unix_server(lambda r, w: _handle(handler, on_close, r, w),
                                             path=path, limit=LINE_LIMIT)
    os.chmod(path, 0o600)
    return server


class AsyncClient:
    """One multiplexed connection. Calls may be issued concurrently; failures raise
    `error_class(kind, detail)`."""
    error_class = EvalrtError

    def __init__(self, reader, writer):
        self._reader = reader
        self._writer = writer
        self._ids = itertools.count(1)
        self._pending = {}
        self._lock = asyncio.Lock()
        self._read_task = asyncio.ensure_future(self._read_loop())

    @classmethod
    async def connect(cls, path):
        reader, writer = await asyncio.open_unix_connection(path, limit=LINE_LIMIT)
        return cls(reader, writer)

    async def _read_loop(self):
        try:
            while True:
                line = await self._reader.readline()
                if not line:
                    break
                message = json.loads(line)
                future = self._pending.pop(message.get('id'), None)
                if future is None or future.done():
                    continue
                if message.get('ok'):
                    future.set_result(message.get('result'))
                else:
                    err = message.get('error') or {}
                    future.set_exception(self.error_class(err.get('kind', 'internal'),
                                                          err.get('detail', '')))
        except (ConnectionError, ValueError, asyncio.CancelledError):
            pass
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(self.error_class('disconnected', 'lost the connection'))
            self._pending.clear()

    async def call(self, op, **fields):
        # Once the read loop has ended nothing will ever answer: a write to a closed transport is
        # dropped silently (asyncio only logs it), so a call made then would wait forever.
        if self._read_task.done():
            raise self.error_class('disconnected', 'the connection is closed')
        rid = next(self._ids)
        future = asyncio.get_running_loop().create_future()
        self._pending[rid] = future
        data = canonical(dict(fields, id=rid, op=op)) + b'\n'
        try:
            async with self._lock:
                self._writer.write(data)
                await self._writer.drain()
        except (ConnectionError, RuntimeError) as exc:
            self._pending.pop(rid, None)
            raise self.error_class('disconnected', str(exc))
        if self._read_task.done() and not future.done():
            self._pending.pop(rid, None)
            raise self.error_class('disconnected', 'the connection closed during the call')
        return await future

    async def close(self):
        self._read_task.cancel()
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:
            pass
        await asyncio.gather(self._read_task, return_exceptions=True)


async def close_server(server, grace=5.0):
    """Stop listening, then wait at most `grace` seconds for open connections. From Python 3.12,
    `Server.wait_closed` waits for every client connection to close, and long-lived clients (an
    optimizer's control connection) never do, so an unbounded wait hangs shutdown."""
    server.close()
    try:
        await asyncio.wait_for(server.wait_closed(), grace)
    except asyncio.TimeoutError:
        pass
