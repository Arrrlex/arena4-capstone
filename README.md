# Capstone for ARENA v4

Full docs coming soon!

## Setting up locally

Make sure uv is installed and run `uv sync`.

## Setting up a new machine

My laptop isn't powerful enough to run Gemma locally, so here's the steps I've been taking to get everything up and running on a vast.ai machine:

1. Make sure the machine is beefy enough
   - For gemma 2b, at least 20gb storage (I've been using 32gb) and 12gb vram (I used an RTX 4090)
2. Add to your SSH config by copying the "Direct ssh connect" command and running:
   ```bash
   uv run add_ssh_host.py vast "ssh -p ...."
   ```
   This will add an ssh host called "vast" to your SSH config. Change "vast" to anything you want this host to be called.
3. Get api keys and put them in `.env` inside `arena4-capstone`
   - From my local checkout of arena4-capstone, I run `scp .env vast:arena4-capstone/`
   - You can read the `Settings` class in [util.py](/src/arena4_capstone/util.py) to see what API keys to get and which are required
4. Configure git
   - Locally run `scp ~/.gitconfig vast:~/`
5. Connect to the machine
6. On the remote machine, install requirements & clone repo using this command:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh && \
   source ~/.bashrc && uv python install 3.12 && \
   git clone https://github.com/Arrrlex/arena4-capstone.git && \
   cd arena4-capstone && \
   uv sync
   ```
7. If using VSCode or Cursor, install Python and Jupyter extensions
