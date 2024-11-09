import os
import sys
from dataclasses import dataclass


@dataclass
class SSHHost:
    host: str
    port: int
    user: str
    host_name: str
    local_port: int
    local_host: str
    remote_port: int

    @classmethod
    def from_args(cls, host: str, ssh_cmd: str):
        cmd, dash_p, port, connection_string, dash_L, fwds = ssh_cmd.split()

        user, host_name = connection_string.split("@")
        local_port, local_host, remote_port = fwds.split(":")

        assert dash_p == "-p"
        assert dash_L == "-L"

        return cls(
            host=host,
            port=int(port),
            user=user,
            host_name=host_name,
            local_port=int(local_port),
            local_host=local_host,
            remote_port=int(remote_port),
        )


def prepend_new_host_to_ssh_config(host: SSHHost, path: str):
    # Read existing content
    try:
        with open(path, "r") as f:
            existing_content = f.read()
    except FileNotFoundError:
        existing_content = ""

    # Create new config
    new_config = f"Host {host.host}\n"
    new_config += f"    HostName {host.host_name}\n"
    new_config += f"    Port {host.port}\n"
    new_config += f"    User {host.user}\n"
    new_config += (
        f"    LocalForward {host.local_port} {host.local_host}:{host.remote_port}\n"
    )

    # Write new config followed by existing content
    with open(path, "w") as f:
        f.write(new_config + "\n" + existing_content)


def remove_host_from_ssh_config(host: str, source_path: str, target_path: str):
    with open(source_path, "r") as f:
        lines = f.readlines()

    with open(target_path, "w") as f:
        in_host = False
        for line in lines:
            if line.strip().startswith("Host " + host):
                in_host = True
            elif line.strip().startswith("Host "):
                in_host = False

            if not in_host:
                f.write(line)


if __name__ == "__main__":
    host = SSHHost.from_args(*sys.argv[1:])
    ssh_config_path = os.path.expanduser("~/.ssh/config")
    ssh_config_path_tmp = "/tmp/ssh_config"

    print("Before removing: ", ssh_config_path)
    print(open(ssh_config_path, "r").read())

    remove_host_from_ssh_config(host.host, ssh_config_path, ssh_config_path_tmp)
    prepend_new_host_to_ssh_config(host, ssh_config_path_tmp)

    print("After modifying: ", ssh_config_path_tmp)
    print(open(ssh_config_path_tmp, "r").read())

    ok = input("Continue? [y/N] ")
    if ok != "y":
        print("Aborting")
        sys.exit(1)
    print(f"Renaming {ssh_config_path_tmp} to {ssh_config_path}")
    os.rename(ssh_config_path_tmp, ssh_config_path)
