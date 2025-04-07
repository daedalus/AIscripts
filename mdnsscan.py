import subprocess
import re
import signal
import sys
from functools import partial

DEBUG = False
DNSSD = "dns-sd"
SCAN = ["_services._dns-sd._udp", "_ipp._tcp"]
STYPE = "Service Type"
DOMAIN = "local."

services = {}
lookup = {}
scanned = set()
hosts = set()

def scan(service, timeout=2):
    cmd = f"{DNSSD} -B '{service}' {DOMAIN}"
    slen = None
    ilen = None
    nlen = None

    def process_line(line):
        nonlocal slen, ilen, nlen
        if slen is None:
            match = re.search(rf"{STYPE}(\s+)", line)
            if match:
                slen = line.find(STYPE)
                nlen = slen + len(STYPE) + len(match.group(1))
                ilen = line.find("Instance Name")
        else:
            service_name = line[slen:nlen].strip()
            instance_name = line[ilen:].strip()

            if service_name.endswith(f"{DOMAIN}"):
                services[instance_name] = service_name
            else:
                lookup[instance_name] = service_name

    run_command(cmd, process_line, timeout)

def lookup_service(instance, service, timeout=1):
    cmd = f"{DNSSD} -L '{instance}' {service} {DOMAIN}"
    
    def process_line(line):
        if DEBUG:
            print(line, file=sys.stderr)
        match = re.search(r"can be reached at (\S+)", line)
        if match:
            host = match.group(1)
            if host not in hosts:
                print(f"{host}:{instance}:{service}")
                hosts.add(host)

    run_command(cmd, process_line, timeout)

def run_command(cmd, callback, timeout=3):
    def alarm_handler(signum, frame):
        raise TimeoutError("Command timed out")

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout)

    try:
        if DEBUG:
            print(f"running {cmd} (alarm {timeout})", file=sys.stderr)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in process.stdout:
            callback(line)
        signal.alarm(0)
    except TimeoutError:
        if DEBUG:
            print(f"Command timed out: {cmd}", file=sys.stderr)
        process.kill()
    finally:
        signal.alarm(0)

def main():
    global DEBUG
    if len(sys.argv) > 1:
        DEBUG = True

    for service in SCAN:
        scan(service)

    for instance, service in lookup.items():
        if instance not in scanned:
            scanned.add(instance)
            lookup_service(instance, service)

if __name__ == "__main__":
    main()
