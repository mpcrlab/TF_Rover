
import struct
import sys


def dump_bytes(bytes):
    for c in bytes:
        sys.stdout.write('%02x ' % ord(c))
    sys.stdout.write('\n')


def bytes_to_int(bytes, offset):
    return struct.unpack('i', bytes[offset:offset + 4])[0]


def bytes_to_uint(bytes, offset):
    return struct.unpack('I', bytes[offset:offset + 4])[0]


def bytes_to_short(bytes, offset):
    return struct.unpack('h', bytes[offset:offset + 2])[0]
