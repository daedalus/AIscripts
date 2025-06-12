import sys
import subprocess

TCP_STATES = {
    '01': 'ESTABLISHED',
    '02': 'SYN_SENT',
    '03': 'SYN_RECV',
    '04': 'FIN_WAIT1',
    '05': 'FIN_WAIT2',
    '06': 'TIME_WAIT',
    '07': 'CLOSE',
    '08': 'CLOSE_WAIT',
    '09': 'LAST_ACK',
    '0A': 'LISTEN',
    '0B': 'CLOSING',
}

def hex_to_ip(hex_str):
    bytes_str = bytearray.fromhex(hex_str)
    return '.'.join(str(b) for b in bytes_str[::-1])

def hex_to_port(hex_str):
    return int(hex_str, 16)

def get_lines(source, use_adb=False):
    if use_adb:
        try:
            output = subprocess.check_output(['adb', 'shell', 'cat', source], stderr=subprocess.DEVNULL)
            return output.decode().splitlines()[1:]  # skip header
        except Exception as e:
            print(f"ADB error reading {source}: {e}")
            return []
    else:
        try:
            with open(source, 'r') as f:
                return f.readlines()[1:]
        except Exception as e:
            print(f"File error reading {source}: {e}")
            return []

def parse_tcp(use_adb=False):
    lines = get_lines('/proc/net/tcp', use_adb)
    print(f"{'L_IP:PORT':<22} {'R_IP:PORT':<22} {'STATE':<13} {'UID':<6} {'INODE':<10}")
    print('-' * 75)

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        local_ip, local_port = parts[1].split(':')
        remote_ip, remote_port = parts[2].split(':')
        state = parts[3]
        uid = parts[7]
        inode = parts[9]

        lip = hex_to_ip(local_ip)
        lport = hex_to_port(local_port)
        rip = hex_to_ip(remote_ip)
        rport = hex_to_port(remote_port)
        state_str = TCP_STATES.get(state, 'UNKNOWN')

        print(f"{lip}:{lport:<16} {rip}:{rport:<16} {state_str:<13} {uid:<6} {inode:<10}")

def parse_udp(use_adb=False):
    lines = get_lines('/proc/net/udp', use_adb)
    print(f"{'L_IP:PORT':<22} {'R_IP:PORT':<22} {'UID':<6} {'INODE':<10}")
    print('-' * 65)

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        local_ip, local_port = parts[1].split(':')
        remote_ip, remote_port = parts[2].split(':')
        uid = parts[7]
        inode = parts[9]

        lip = hex_to_ip(local_ip)
        lport = hex_to_port(local_port)
        rip = hex_to_ip(remote_ip)
        rport = hex_to_port(remote_port)

        print(f"{lip}:{lport:<16} {rip}:{rport:<16} {uid:<6} {inode:<10}")

def parse_unix(use_adb=False):
    lines = get_lines('/proc/net/unix', use_adb)
    print(f"{'Type':<7} {'State':<6} {'Inode':<10} {'Path'}")
    print('-' * 60)

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 7:
            sock_type = parts[4]
            state = parts[5]
            inode = parts[6]
            path = parts[7] if len(parts) > 7 else ''
            print(f"{sock_type:<7} {state:<6} {inode:<10} {path}")

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {'tcp', 'udp', 'unix'}:
        print("Usage: python netstat_android.py [tcp|udp|unix] [--adb]")
        sys.exit(1)

    mode = sys.argv[1]
    use_adb = '--adb' in sys.argv

    if mode == 'tcp':
        parse_tcp(use_adb)
    elif mode == 'udp':
        parse_udp(use_adb)
    elif mode == 'unix':
        parse_unix(use_adb)

