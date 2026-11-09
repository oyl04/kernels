"""Check the private address map supplied for the Ascend cluster."""

from concurrent.futures import ThreadPoolExecutor
from getpass import getpass
from sys import argv

from run_remote_ssh import connect, read_a5_password


HOSTS = dict(enumerate((100, 164, 122, 16, 112, 242, 155, 185), 1))


def via(jump, suffix, password, port=22):
    host = f"192.168.0.{suffix}"
    channel = jump.get_transport().open_channel(
        "direct-tcpip", (host, port), ("127.0.0.1", 0), timeout=10
    )
    try:
        return connect(host, port, "root", password, sock=channel, timeout=12)
    except Exception:
        channel.close()
        raise


def scan(jump, password, skip=None, port=22):
    def inspect(item):
        number, suffix = item
        label = f"{number:02d} (192.168.0.{suffix})"
        try:
            with via(jump, suffix, password, port) as target:
                _, stdout, _ = target.exec_command("hostname; whoami", timeout=10)
                identity = stdout.read().decode().splitlines()
                if (len(identity) != 2 or identity[1] != "root"
                        or identity[0].split(".")[0] != f"hyperinstance-cfdtc-{number:05d}"):
                    return f"{label}: identity mismatch {identity!r}"
                _, stdout, stderr = target.exec_command("npu-smi info", timeout=15)
                output = stdout.read().decode(errors="replace")
                error = stderr.read().decode(errors="replace")
                return f"{label}: npu-smi exit {stdout.channel.recv_exit_status()}\n{output}{error}"
        except Exception as exc:
            return f"{label}: {type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(inspect, (item for item in HOSTS.items() if item[0] != skip)):
            print(result, flush=True)


def main():
    password = getpass("Bastion and target password: ")
    with connect("193.112.219.133", 5002, "root", password, timeout=20) as bastion:
        port = 5002 if "--port-5002" in argv else 22
        use_existing = "--existing-03" in argv or port == 5002
        if "--existing-03" not in argv:
            print("=== Through bastion ===", flush=True)
            scan(bastion, password, port=port)
        print("=== Through server 03 ===", flush=True)
        try:
            if use_existing:
                address = "116.63.178.251"
                jump_password = read_a5_password(bastion)
                channel = bastion.get_transport().open_channel(
                    "direct-tcpip", (address, 22), ("127.0.0.1", 0), timeout=10
                )
                server03 = connect(address, 22, "root", jump_password, sock=channel, timeout=12)
            else:
                server03 = via(bastion, HOSTS[3], password)
            with server03:
                _, stdout, _ = server03.exec_command("hostname", timeout=10)
                hostname = stdout.read().decode().strip().split(".")[0]
                if hostname != "hyperinstance-cfdtc-00003":
                    raise RuntimeError(f"Unexpected jump hostname: {hostname}")
                scan(server03, password, skip=None if use_existing else 3, port=port)
        except Exception as exc:
            print(f"Server 03 jump failed: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
