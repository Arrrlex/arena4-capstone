# Capstone for ARENA v4

Full docs coming soon!

## Setting up a new machine

My laptop isn't powerful enough to run Gemma locally, so here's the steps I've been taking to get everything up and running on a vast.ai machine:

1. Make sure the machine is beefy enough
  - For gemma 2b, at least 20gb storage (I've been using 32gb) and 12gb vram (I used an RTX 4090)
2. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then `source ~/.bashrc`
3. Install python: `uv python install 3.12`
4. Clone repo: `git clone https://github.com/Arrrlex/arena4-capstone.git`
5. Install dependencies: `cd ~/arena4-capstone && uv sync`
7. Get api keys and put them in `.env` inside `arena4-capstone`
  - From my local checkout of arena4-capstone, I run `scp .env vast:arena4-capstone/`
8. Configure git
  1. Locally run `scp ~/.gitconfig vast:~/`
  2. On remote machine, run `git config credential.helper store`
  3. On remote machine run `echo <username>:<PAT>@github.com > ~/.git-credentials`, using your real github username and PAT
6. If using VSCode or Cursor, install Python and Jupyter extensions
