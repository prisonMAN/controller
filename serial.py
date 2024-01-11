import serial

class SerialPort:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def open(self):
        self.ser = serial.Serial(self.port, self.baudrate)

    def send_data(self, data):
        if self.ser is not None and self.ser.is_open:
            self.ser.write(data)

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()