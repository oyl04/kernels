#!/usr/bin/env python3
"""Run a single command on a private target through the approved bastion."""

import argparse
import getpass
import sys
import time

import paramiko


PRIVATE_HOSTS = {
    "01": "192.168.0.221",
    "02": "192.168.0.8",
    "04": "192.168.0.56",
    "05": "192.168.0.196",
    "06": "192.168.0.187",
    "07": "192.168.0.60",
    "08": "192.168.0.26",
}

PUBLIC_HOSTS = {
    "01": "116.63.189.104",
    "02": "139.9.249.211",
    "03": "116.63.178.251",
    "04": "139.9.157.42",
    "05": "122.9.130.212",
    "06": "122.9.154.45",
    "07": "116.63.188.7",
    "08": "122.9.148.227",
}


def connect(host, port, user, password, sock=None, timeout=10):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=user,
        password=password,
        sock=sock,
        timeout=timeout,
        banner_timeout=timeout,
        auth_timeout=timeout,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def connect_with_retries(host, port, user, password, timeout, attempts):
    for attempt in range(1, attempts + 1):
        try:
            return connect(host, port, user, password, timeout=timeout)
        except (OSError, paramiko.SSHException) as exc:
            if attempt == attempts:
                raise
            print(
                f"Connection attempt {attempt}/{attempts} failed: {exc}; retrying...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(2)


def read_a5_password(bastion):
    _, stdout, stderr = bastion.exec_command(
        "sed -n '4{s/^[^:=]*[:=][[:space:]]*//;p;}' /root/info/a5.txt",
        timeout=10,
    )
    password = stdout.read().decode().strip()
    error = stderr.read().decode().strip()
    if stdout.channel.recv_exit_status() != 0 or not password:
        raise RuntimeError(error or "could not read the A5 password")
    return password


def probe_private_hosts(bastion, command="hostname"):
    a5_password = read_a5_password(bastion)
    candidate_password = getpass.getpass("Candidate private-host password: ")
    hop_channel = bastion.get_transport().open_channel(
        "direct-tcpip", ("116.63.178.251", 22), ("127.0.0.1", 0), timeout=10
    )
    server03 = connect(
        "116.63.178.251", 22, "root", a5_password, sock=hop_channel
    )
    try:
        for label, host in PRIVATE_HOSTS.items():
            target = None
            channel = None
            try:
                channel = server03.get_transport().open_channel(
                    "direct-tcpip", (host, 22), ("127.0.0.1", 0), timeout=10
                )
                target = connect(host, 22, "root", candidate_password, sock=channel)
                _, stdout, stderr = target.exec_command(command, timeout=30)
                hostname = stdout.read().decode(errors="replace").strip()
                error = stderr.read().decode(errors="replace").strip()
                status = stdout.channel.recv_exit_status()
                if status == 0 and hostname:
                    print(f"{label} {host}: connected")
                    print(hostname)
                else:
                    print(f"{label} {host}: command failed ({error or status})")
            except Exception as exc:
                print(f"{label} {host}: failed ({type(exc).__name__}: {exc})")
            finally:
                if target:
                    target.close()
                elif channel:
                    channel.close()
    finally:
        server03.close()


def probe_public_hosts(bastion, command="hostname"):
    candidate_password = getpass.getpass("Candidate public-host password: ")
    for label, host in PUBLIC_HOSTS.items():
        target = None
        channel = None
        try:
            channel = bastion.get_transport().open_channel(
                "direct-tcpip", (host, 22), ("127.0.0.1", 0), timeout=10
            )
            target = connect(host, 22, "root", candidate_password, sock=channel)
            _, stdout, stderr = target.exec_command(command, timeout=30)
            hostname = stdout.read().decode(errors="replace").strip()
            error = stderr.read().decode(errors="replace").strip()
            status = stdout.channel.recv_exit_status()
            if status == 0 and hostname:
                print(f"=== {label} {host} ===")
                print(hostname)
            else:
                print(f"{label} {host}: command failed ({error or status})")
        except Exception as exc:
            print(f"{label} {host}: failed ({type(exc).__name__}: {exc})")
        finally:
            if target:
                target.close()
            elif channel:
                channel.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--user", default="root")
    parser.add_argument("--command")
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--connect-attempts", type=int, default=1)
    parser.add_argument("--upload-local")
    parser.add_argument("--upload-remote")
    parser.add_argument("--bastion-command")
    parser.add_argument("--probe-private-hosts", action="store_true")
    parser.add_argument("--probe-public-hosts", action="store_true")
    parser.add_argument("--probe-command", default="hostname")
    args = parser.parse_args()
    modes = sum(
        bool(value)
        for value in (
            args.host,
            args.bastion_command,
            args.probe_private_hosts,
            args.probe_public_hosts,
        )
    )
    if modes != 1:
        parser.error(
            "provide exactly one connection mode"
        )
    upload_requested = bool(args.upload_local or args.upload_remote)
    if bool(args.upload_local) != bool(args.upload_remote):
        parser.error("--upload-local and --upload-remote must be provided together")
    if args.host and bool(args.command) == upload_requested:
        parser.error("with --host, provide exactly one command or upload pair")

    jump_password = getpass.getpass("Bastion password: ")
    bastion = target = None
    try:
        bastion = connect_with_retries(
            "193.112.219.133",
            5002,
            "root",
            jump_password,
            args.connect_timeout,
            args.connect_attempts,
        )
        if args.probe_private_hosts:
            probe_private_hosts(bastion, args.probe_command)
            return 0
        if args.probe_public_hosts:
            probe_public_hosts(bastion, args.probe_command)
            return 0
        if args.bastion_command:
            _, stdout, stderr = bastion.exec_command(args.bastion_command, timeout=30)
            sys.stdout.write(stdout.read().decode(errors="replace"))
            sys.stderr.write(stderr.read().decode(errors="replace"))
            return stdout.channel.recv_exit_status()
        target_password = getpass.getpass("Target password: ")
        channel = bastion.get_transport().open_channel(
            "direct-tcpip", (args.host, 22), ("127.0.0.1", 0), timeout=10
        )
        target = connect(
            args.host,
            22,
            args.user,
            target_password,
            sock=channel,
            timeout=args.connect_timeout,
        )
        if upload_requested:
            sftp = target.open_sftp()
            try:
                sftp.put(args.upload_local, args.upload_remote)
            finally:
                sftp.close()
            return 0
        _, stdout, stderr = target.exec_command(args.command, timeout=30)
        sys.stdout.write(stdout.read().decode(errors="replace"))
        sys.stderr.write(stderr.read().decode(errors="replace"))
        return stdout.channel.recv_exit_status()
    finally:
        if target:
            target.close()
        if bastion:
            bastion.close()


if __name__ == "__main__":
    raise SystemExit(main())
