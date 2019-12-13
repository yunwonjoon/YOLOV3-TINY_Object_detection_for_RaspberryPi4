import serial
ser = serial.Serial(port='COM8',baudrate=9600,)
while True:
    if ser.readable():
        res=ser.readline().decode()
        if res=="a":
            ser.write("a".encode('ascii'))
        else:
            ser.write("b".encode('ascii'))
        print(res)