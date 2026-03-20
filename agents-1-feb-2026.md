# Local Context

## Identity
- Folder: 003-Qwen3-VL
- Depth: 1
- Generated: 2026-02-01

## Notes
- Folder created or updated after 2026-01-24; add local notes as needed.

## Local Setup Scripts (2026-01-13)
- `uiuc_vpn_cli.sh`: UIUC VPN (OpenConnect) + SSH/RDP/Remmina/AnyDesk/Cisco client launchers.
- `add_student_key.sh`: Add a public key to a target user's `authorized_keys` (optional `--enable-ssh`).
- `ngc_setup.sh`: `ngc config set` + `docker login nvcr.io` with hidden key input.
- `ngc_docker_install.sh`: installs Docker + NGC CLI (with prereqs).
- `nvidia_container_toolkit_install.sh`: installs NVIDIA Container Toolkit and configures Docker runtime.
- `docker_runtime_check.sh`: verifies Docker + NVIDIA runtime configuration without pulling images.
- `docker_gpu_test.sh`: pulls CUDA image and runs `nvidia-smi` in a container.
- `docker-compose.gpu-test.yml` + `docker_gpu_compose_test.sh`: compose-based GPU test runner.
- `vscode_tunnel.sh`: Launch VS Code Remote Tunnels (foreground or background).

## Reference
- See agents.md at the repo root for workflows, pipelines, and directory map.
