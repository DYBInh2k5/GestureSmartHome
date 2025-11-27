import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 9999

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto("LIGHT_ON".encode(), (UDP_IP, UDP_PORT))
print("Sent LIGHT_ON")
