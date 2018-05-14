import threading
import socket
import time
import numpy as np
import cv2

from rover.blowfish import Blowfish
from rover.adpcm import decodeADPCMToPCM
from rover.byteutils import *

# Base class for handling sockets, encryption, and movement
class Rover:
    def __init__(self):
        """ Creates a Rover object that you can communicate with.
        """

        self.HOST = '192.168.1.100'
        self.PORT = 80

        TARGET_ID = 'AC13'
        TARGET_PASSWORD = 'AC13'

        self.TREAD_DELAY_SEC = 0.05
        self.KEEPALIVE_PERIOD_SEC = 60

        # Create command socket connection to Rover
        self.commandsock = self._new_socket()

        # Send login request with four arbitrary numbers
        self._send_command_int_request(0, [0, 0, 0, 0])

        # Get login reply
        reply = self._receive_a_command_reply_from_rover(82)

        # Extract Blowfish key from camera ID in reply
        camera_ID = reply[25:37].decode('utf-8')
        key = TARGET_ID + ':' + camera_ID + '-save-private:' + TARGET_PASSWORD

        # Extract Blowfish inputs from rest of reply
        l = bytes_to_int(reply, 66)
        r1 = bytes_to_int(reply, 70)
        l2 = bytes_to_int(reply, 74)
        r2 = bytes_to_int(reply, 78)

        # Make Blowfish cipher from key
        bf = _RoverBlowfish(key)

        # Encrypt inputs from reply
        l, r1 = bf.encrypt(l, r1)
        l2, r2 = bf.encrypt(l2, r2)

        # Send encrypted reply to Rover
        self._send_command_int_request(2, [l, r1, l2, r2])

        # Ignore reply from Rover
        self._receive_a_command_reply_from_rover(26)

        # Start timer task for keep-alive message every 60 seconds
        self._start_keep_rover_alive_task()

        # Setup vertical camera controller
        self.cameraVertical = _RoverCamera(self, 1)

        # Send video-start request
        self._send_command_int_request(4, [1])

        # Get reply from Rover
        reply = self._receive_a_command_reply_from_rover(29)

        # Create media socket connection to Rover
        self.mediasock = self._new_socket()

        # Send video-start request based on last four bytes of reply
        self._send_a_request(self.mediasock, 'V', 0, 4, map(ord, reply[25:]))

        # Send audio-start request
        self._send_command_byte_request(8, [1])

        # Ignore audio-start reply
        self._receive_a_command_reply_from_rover(25)

        # Receive images on another thread until closed
        self.is_active = True
        self.reader_thread = _MediaThread(self)
        self.reader_thread.start()

        # Set up treads
        self.leftTread = _RoverTread(self, 4)
        self.rightTread = _RoverTread(self, 1)

    def close(self):
        """ Closes off communication with Rover.
        """

        self.keep_a_live_timer.cancel()

        self.is_active = False
        self.commandsock.close()

        if self.mediasock:
            self.mediasock.close()

        # Stop moving treads
        self.set_wheel_treads(0, 0)

    def turn_stealth_on(self):
        """ Turns on stealth mode (infrared).
        """
        self._send_camera_request(94)

    def turn_stealth_off(self):
        """ Turns off stealth mode (infrared).
        """
        self._send_camera_request(95)

    def move_camera_in_vertical_direction(self, where):
        """ Moves the camera up or down, or stops moving it.  A nonzero value for the
            where parameter causes the camera to move up (+) or down (-).  A
            zero value stops the camera from moving.
        """
        self.cameraVertical.move(where)

    def _start_keep_rover_alive_task(self, ):
        self._send_command_byte_request(255)
        self.keep_a_live_timer = \
            threading.Timer(self.KEEPALIVE_PERIOD_SEC, self._start_keep_rover_alive_task, [])
        self.keep_a_live_timer.start()

    def _send_command_byte_request(self, request_id, bytes_request=None):
        if not bytes_request:
            bytes_request = []
        self._send_a_command_request(request_id, len(bytes_request), bytes_request)

    def _send_command_int_request(self, request_id, intervals):
        byte_value = []
        for val in intervals:
            for c in struct.pack('I', val):
                byte_value.append(ord(c))
        self._send_a_command_request(request_id, 4 * len(intervals), byte_value)

    def _send_a_command_request(self, id_command_request, n, contents):
        self._send_a_request(self.commandsock, 'O', id_command_request, n, contents)

    def _send_a_request(self, sock, c, id_request, n, contents):
        bytes_request = [ord('M'), ord('O'), ord('_'), ord(c), id_request,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n, 0, 0, 0, 0, 0, 0, 0]
        bytes_request.extend(contents)
        request = ''.join(map(chr, bytes_request))
        sock.send(request)

    def _receive_a_command_reply_from_rover(self, count):
        reply = self.commandsock.recv(count)
        return reply

    def _new_socket(self):
        sock = socket.socket()
        sock.connect((self.HOST, self.PORT))
        return sock

    def _send_control_request_to_rover(self, a, b):
        self._send_command_byte_request(250, [a, b])

    # 2.0 overrides:

    def _send_camera_request(self, request):
        self._send_command_byte_request(14, [request])

        #def __init__(self):
        #Rover.__init__(self)

        # Set up treads
        #self.leftTread = _RoverTread(self, 4)
        #self.rightTread = _RoverTread(self, 1)

    def get_battery_percentage(self):
        """ Returns percentage of battery remaining.
        """
        self._send_command_byte_request(251)
        reply = self._receive_a_command_reply_from_rover(32)
        return 15 * ord(reply[23])

    def set_wheel_treads(self, left, right):
        """ Sets the speed of the left and right treads (wheels).  + = forward;
        - = backward; 0 = stop. Values should be in [-1..+1].
        """
        currTime = time.time()

        self.leftTread.update(left)
        self.rightTread.update(right)

    def turn_the_lights_on(self):
        """ Turns the headlights and taillights on.
        """
        self._set_the_lights_on_or_off(8)

    def turn_the_lights_off(self):
        """ Turns the headlights and taillights off.
        """
        self._set_the_lights_on_or_off(9)

    def _set_the_lights_on_or_off(self, on_or_off):
        self._send_control_request_to_rover(on_or_off, 0)

    def process_video_from_rover(self, jpegbytes, timestamp_10msec):
        array_of_bytes = np.fromstring(jpegbytes, np.uint8)
        self.image = cv2.imdecode(array_of_bytes, flags=3)
        k = cv2.waitKey(1) & 0xFF
        return self.image

    def process_audio_from_rover(self, pcmsamples, timestamp_10msec):
        """ Processes a block of 320 PCM audio samples streamed from Rover.
            Audio is sampled at 8192 Hz and quantized to +/- 2^15.
            Default method is a no-op; subclass and override to do something
            interesting.
        """
        pass

    def _spin_rover_wheels(self, wheel_direction, speed):
        # 1: Right, forward
        # 2: Right, backward
        # 4: Left, forward
        # 5: Left, backward
        self._send_control_request_to_rover(wheel_direction, speed)

        # "Private" classes ===========================================================


# A special Blowfish variant with P-arrays set to zero instead of digits of Pi
class _RoverBlowfish(Blowfish):
    def __init__(self, key):
        Blowfish.__init__(self, key)
        ORIG_P = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._keygen(key, ORIG_P)


# A thread for reading streaming media from the Rover
class _MediaThread(threading.Thread):
    def __init__(self, rover):

        threading.Thread.__init__(self)

        self.rover = rover
        self.buffer_size = 1024

    def run(self):

        # Accumulates media bytes
        media_bytes = ''

        # Starts True; set to False by Rover.close()
        while self.rover.is_active:

            # Grab bytes from rover, halting on failure
            try:
                buf = self.rover.mediasock.recv(self.buffer_size)
            except:
                break

            # Do we have a media frame start?
            k = buf.find('MO_V')

            # Yes
            if k >= 0:

                # Already have media bytes?
                if len(media_bytes) > 0:

                    # Yes: add to media bytes up through start of new
                    media_bytes += buf[0:k]

                    # Both video and audio messages are time-stamped in 10msec units
                    timestamp = bytes_to_uint(media_bytes, 23)

                    # Video bytes: call processing routine
                    if ord(media_bytes[4]) == 1:
                        self.rover.process_video_from_rover(media_bytes[36:], timestamp)

                    # Audio bytes: call processing routine
                    else:
                        audio_size = bytes_to_uint(media_bytes, 36)
                        sample_audio_size = 40 + audio_size
                        offset = bytes_to_short(media_bytes, sample_audio_size)
                        index = ord(media_bytes[sample_audio_size + 2])
                        pcmsamples = decodeADPCMToPCM(media_bytes[40:sample_audio_size], offset, index)
                        self.rover.process_audio_from_rover(pcmsamples, timestamp)

                        # Start over with new bytes
                    media_bytes = buf[k:]

                # No media bytes yet: start with new bytes
                else:
                    media_bytes = buf[k:]

            # No: accumulate media bytes
            else:

                media_bytes += buf


class _RoverTread(object):
    def __init__(self, rover, index):

        self.rover = rover
        self.index = index
        self.isMoving = False
        self.startTime = 0

    def update(self, value):

        if value == 0:
            if self.isMoving:
                self.rover._spin_rover_wheels(self.index, 0)
                self.isMoving = False
        else:
            if value > 0:
                wheel = self.index
            else:
                wheel = self.index + 1
            current_run_time = time.time()
            if (current_run_time - self.startTime) > self.rover.TREAD_DELAY_SEC:
                self.startTime = current_run_time
                self.rover._spin_rover_wheels(wheel, int(round(abs(value) * 10)))
                self.isMoving = True


class _RoverCamera(object):
    def __init__(self, rover, stop_rover_command):

        self.rover = rover
        self.stop_rover_command = stop_rover_command
        self.isMoving = False

    def move(self, where):

        if where == 0:
            if self.isMoving:
                self.rover._send_camera_request(self.stop_rover_command)
                self.isMoving = False
        elif not self.isMoving:
            if where == 1:
                self.rover._send_camera_request(self.stop_rover_command - 1)
            else:
                self.rover._send_camera_request(self.stop_rover_command + 1)
            self.isMoving = True
