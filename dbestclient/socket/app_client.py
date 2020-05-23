#!/usr/bin/env python3

import selectors
import socket
# import sys
import traceback
from datetime import datetime

from dbestclient.socket import libclient

verbose = False


def create_request(action, value):
    if action == "select":
        return dict(
            type="text/json",
            encoding="utf-8",
            content=dict(action=action, value=value),
        )
    elif action == "search":
        return dict(
            type="text/json",
            encoding="utf-8",
            content=dict(action=action, value=value),
        )
    else:
        return dict(
            type="binary/custom-client-binary-type",
            encoding="binary",
            content=bytes(action + value, encoding="utf-8"),
        )


def start_connection(host, port, request):
    addr = (host, port)
    if verbose:
        print("starting connection to", addr)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel = selectors.DefaultSelector()
    message = libclient.Message(sel, sock, addr, request)
    sel.register(sock, events, data=message)
    return sel


def run(host, port, actions, action_value):
    t1 = datetime.now()
    action, value = actions, action_value  # 'search', 'ring'
    request = create_request(action, value)
    sel = start_connection(host, port, request)
    # sel = selectors.DefaultSelector()
    result = None

    try:
        while True:
            events = sel.select(timeout=1)
            for key, mask in events:
                message = key.data
                try:
                    # t1 = datetime.now()
                    reslt = message.process_events(mask)
                    if reslt is not None:
                        result = reslt
                    # print("result,", result)
                    # t2 = datetime.now()
                    # print("time cost is ", (t2-t1).total_seconds())
                except Exception:
                    print(
                        "main: error: exception for",
                        f"{message.addr}:\n{traceback.format_exc()}",
                    )
                    message.close()
            # Check for a socket being monitored to continue.
            if not sel.get_map():
                break
    except KeyboardInterrupt:
        print("caught keyboard interrupt, exiting")
    finally:
        sel.close()
        # print("result,", result)
        if verbose:
            t2 = datetime.now()
            print("time cost is ", (t2-t1).total_seconds())
        return result
