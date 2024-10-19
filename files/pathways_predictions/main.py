#!/usr/bin/python3
# ECE 498B 2022
# Ben Chapman-Kish

# Usage: `python3 main.py engine [-v] [port]` OR `python3 main.py api [-t (ssh_passthrough_user@)ssh_passthrough_host] [host(:port) ...]``

from __future__ import annotations
from shared.base import Config
import sys, logging

# TODO: Use argparse to set debug_server, verbose_training, engine_server_addrs, ssh tunnel...

# NOTE: The use of SSH tunnels for this project is required due to the configuration of the ECE department's GPU servers on the network.
def establish_ssh_tunnel(local_port: int, remote_host: str, remote_port: int, intermediate_user: str | None, intermediate_host: str):
    import subprocess

    if intermediate_user is None:
        import getpass
        intermediate_user = getpass.getuser()

    # KNOWN ISSUE: If the SSH tunnel is closed at any point during operation, it can't be reopened without restarting the program
    subprocess.Popen(["ssh", "-L", f"{local_port}:{remote_host}:{remote_port}", f"{intermediate_user}@{intermediate_host}"], stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    sys.stdout.write(f"Established ssh tunnel from localhost:{local_port} to {remote_host}:{remote_port} via {intermediate_user}@{intermediate_host}\n")

if __name__ == "__main__":
    assert(len(sys.argv) > 1)
    assert((mode := sys.argv[1]) in ("api", "engine"))
    
    if mode == "api":
        if len(sys.argv) > 2:
            engine_server_addrs: list[Config.ServerAddr] = []
            next_local_port = Config.INITIAL_LOCAL_PORT

            if sys.argv[2] == "-t":
                ssh_tunnel = True
                intermediate_server_addr = sys.argv[3]
                server_addrs = sys.argv[4:]

            else:
                ssh_tunnel = False
                intermediate_server_addr = ""
                server_addrs = sys.argv[2:]

            for server_addr in server_addrs:
                if ':' in server_addr:
                    host, port = server_addr.split(':')
                else:
                    host, port = server_addr, Config.DEFAULT_ENGINE_PORT
                
                port = int(port)

                if ssh_tunnel:
                    if '@' in intermediate_server_addr:
                        intermediate_user, intermediate_host = intermediate_server_addr.split('@')
                    else:
                        intermediate_user, intermediate_host = None, intermediate_server_addr
                    
                    establish_ssh_tunnel(next_local_port, host, port, intermediate_user, intermediate_host)
                    
                    engine_server_addrs.append(Config.ServerAddr(host, port, ssh_tunnel, next_local_port, intermediate_host))
                    next_local_port += 1
                
                else:
                    engine_server_addrs.append(Config.ServerAddr(host, port))
            
            Config.instantiate(
                debug_server=False,
                engine_server_addrs=engine_server_addrs,
                fake_job_ratings_per_user_range=(5,11),
                fake_course_ratings_per_user_range=(0,1),
                fake_ratings_per_user_probability=0.3,
                log_level=logging.DEBUG
            )
        
        else:
            Config.instantiate(
                debug_server=False,
                fake_job_ratings_per_user_range=(5,6),
                fake_course_ratings_per_user_range=(0,1),
                fake_ratings_per_user_probability=1.0,
                log_level=logging.DEBUG
            )

        from api.api import main
        main()
    
    elif mode == "engine":
        port = Config.DEFAULT_ENGINE_PORT
        verbose = False

        if len(sys.argv) > 2:
            if sys.argv[2] == '-v':
                verbose = True

                if len(sys.argv) > 3:
                    port = int(sys.argv[3])
            else:
                port = int(sys.argv[2])

        Config.instantiate(
            debug_server=False,
            verbose_training=verbose,
            engine_port=port,
            log_level=logging.INFO
        )

        from engine.server import main
        main()
