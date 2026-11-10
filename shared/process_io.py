"""UTF-8 logs for Windows GUI and Python worker processes."""
import codecs
import os
import sys


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace', line_buffering=True)


def worker_environment():
    environment = os.environ.copy()
    environment['PYTHONIOENCODING'] = 'utf-8'
    environment['PYTHONUTF8'] = '1'
    return environment


def utf8_decoder():
    # A pipe read may split a multi-byte character between two Qt signals.
    return codecs.getincrementaldecoder('utf-8')(errors='replace')
