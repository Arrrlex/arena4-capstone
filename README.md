# Capstone for ARENA v4

Full docs coming soon!

## Setting up locally

1. Make sure [uv is installed](https://docs.astral.sh/uv/) and run `uv sync`.
2. Obtain any environment variables needed (You can read the `Settings` class in [util.py](/src/arena4_capstone/util.py#L23) to see all of them) and put them in `arena4-capstone/.env`.

## Setting up a new machine

My laptop isn't powerful enough to run Gemma locally, so here's the steps I've been taking to get everything up and running on a vast.ai machine:

1. Make sure the machine is beefy enough
   - For gemma 2b, at least 20gb storage (I've been using 32gb) and 12gb vram (I used an RTX 4090)
2. Add to your SSH config by copying the "Direct ssh connect" command and running:
   ```bash
   uv run add_ssh_host.py vast "ssh -p ...."
   ```
   This will add an ssh host called "vast" to your SSH config. Change "vast" to anything you want this host to be called.
3. On the remote machine, install requirements & clone repo using this command:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh && \
   source ~/.bashrc && uv python install 3.12 && \
   git clone https://github.com/Arrrlex/arena4-capstone.git && \
   cd arena4-capstone && \
   uv sync
   ```
4. If using VSCode or Cursor, ensure Python and Jupyter extensions are installed. You can do this manually, or you can do it once by adding the following snippet to your `settings.json`:
   ```json
   "remote.SSH.defaultExtensions": [
      "ms-toolsai.jupyter",
      "ms-python.python"
   ]
   ```
5. Get any secrets and configuration from your local machine to the remote machine. Locally, run this command:
   ```bash
   scp .env vast:arena4-capstone/ && scp ~/.gitconfig vast:~/
   ```
