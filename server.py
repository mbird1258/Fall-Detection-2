import socket

ServerIp = socket.gethostname()

Socket = socket.socket()
Socket.bind((ServerIp, 8000))
Socket.listen()

print(f"ServerIp: {ServerIp}")

def main():
    Conn, Addr = Socket.accept()
    print("\n\nStart\n\n")

    try:
        while True:
            flag = Conn.recv(4096)

            if flag == b"a":
                print("Signal  gotten")
    except ConnectionResetError as e:
        print(e)
        main()

main()
print("\n\nFin\n\n")