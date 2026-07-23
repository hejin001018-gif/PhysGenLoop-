from __future__ import annotations

import argparse
import os
import posixpath
import stat
import sys
import threading
import time
from pathlib import Path

import paramiko

HOST = os.environ.get("PHYS_SERVER_HOST", "px-cloud2.matpool.com")
PORT = int(os.environ.get("PHYS_SERVER_PORT", "28404"))
USER = "root"
PASSWORD = os.environ.get("PHYS_SERVER_PASSWORD", r"#zJSCR]IP%BK+Ejw")
REMOTE_ROOT = "/root/PhysGenLoop-"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=HOST,
        port=PORT,
        username=USER,
        password=PASSWORD,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def command_shell(client: paramiko.SSHClient, command: str) -> int:
    stdin, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    return stdout.channel.recv_exit_status()


def interactive_shell(client: paramiko.SSHClient) -> int:
    channel = client.invoke_shell()
    channel.send(f"cd {REMOTE_ROOT}\n")

    stop = threading.Event()

    def _reader() -> None:
        while not stop.is_set():
            if channel.recv_ready():
                data = channel.recv(4096)
                if not data:
                    break
                sys.stdout.write(data.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            else:
                time.sleep(0.05)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    try:
        while True:
            try:
                line = input()
            except EOFError:
                break
            channel.send(line + "\n")
    finally:
        stop.set()
        try:
            channel.send("exit\n")
        except Exception:
            pass
        thread.join(timeout=1)
        channel.close()
    return 0


def put_file(client: paramiko.SSHClient, local_path: str, remote_path: str) -> int:
    sftp = client.open_sftp()
    try:
        remote_dir = posixpath.dirname(remote_path)
        parts = remote_dir.split("/")
        current = ""
        for part in parts:
            if not part:
                continue
            current += "/" + part
            try:
                sftp.stat(current)
            except FileNotFoundError:
                sftp.mkdir(current)
        sftp.put(local_path, remote_path)
    finally:
        sftp.close()
    return 0


def sync_dir(client: paramiko.SSHClient, local_dir: str, remote_dir: str) -> int:
    sftp = client.open_sftp()
    try:
        for root, _, files in os.walk(local_dir):
            for name in files:
                local_path = os.path.join(root, name)
                relative = os.path.relpath(local_path, local_dir).replace("\\", "/")
                target = posixpath.join(remote_dir, relative)
                target_dir = posixpath.dirname(target)
                parts = target_dir.split("/")
                current = ""
                for part in parts:
                    if not part:
                        continue
                    current += "/" + part
                    try:
                        sftp.stat(current)
                    except FileNotFoundError:
                        sftp.mkdir(current)
                sftp.put(local_path, target)
                mode = os.stat(local_path).st_mode
                if mode & stat.S_IXUSR:
                    sftp.chmod(target, 0o755)
    finally:
        sftp.close()
    return 0


def get_file(client: paramiko.SSHClient, remote_path: str, local_path: str) -> int:
    sftp = client.open_sftp()
    try:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_path, local_path)
    finally:
        sftp.close()
    return 0


def pull_dir(client: paramiko.SSHClient, remote_dir: str, local_dir: str) -> int:
    sftp = client.open_sftp()
    try:
        local_root = Path(local_dir)
        local_root.mkdir(parents=True, exist_ok=True)
        stack = [(remote_dir.rstrip("/"), local_root)]
        while stack:
            current_remote, current_local = stack.pop()
            current_local.mkdir(parents=True, exist_ok=True)
            for entry in sftp.listdir_attr(current_remote):
                remote_path = posixpath.join(current_remote, entry.filename)
                local_path = current_local / entry.filename
                if stat.S_ISDIR(entry.st_mode):
                    stack.append((remote_path, local_path))
                elif stat.S_ISREG(entry.st_mode):
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(remote_path, str(local_path))
    finally:
        sftp.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PhysGenLoop 远程 SSH 工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("shell", help="打开交互式远程 shell")

    exec_parser = subparsers.add_parser("exec", help="执行远程命令")
    exec_parser.add_argument("remote_command", help="要执行的远程命令")

    put_parser = subparsers.add_parser("put", help="上传单个文件")
    put_parser.add_argument("local_path")
    put_parser.add_argument("remote_path")

    get_parser = subparsers.add_parser("get", help="下载单个文件")
    get_parser.add_argument("remote_path")
    get_parser.add_argument("local_path")

    sync_parser = subparsers.add_parser("sync", help="同步目录到服务器")
    sync_parser.add_argument("local_dir")
    sync_parser.add_argument("remote_dir", nargs="?", default=REMOTE_ROOT)

    pull_parser = subparsers.add_parser("pull", help="从服务器下载目录")
    pull_parser.add_argument("local_dir")
    pull_parser.add_argument("remote_dir", nargs="?", default=REMOTE_ROOT)

    args = parser.parse_args()
    client = connect()
    try:
        if args.command == "shell":
            return interactive_shell(client)
        if args.command == "exec":
            return command_shell(client, args.remote_command)
        if args.command == "put":
            return put_file(client, args.local_path, args.remote_path)
        if args.command == "get":
            return get_file(client, args.remote_path, args.local_path)
        if args.command == "sync":
            return sync_dir(client, args.local_dir, args.remote_dir)
        if args.command == "pull":
            return pull_dir(client, args.remote_dir, args.local_dir)
        raise ValueError(args.command)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
