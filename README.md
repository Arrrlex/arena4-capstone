# Capstone for ARENA v4

Full docs coming soon!

## Setting up a new machine

1. Make sure the machine is beefy enough:
    1. For gemma 9b, at least 24gb storage and >24gb vram (not sure how much needed)
    2. For gemma 2b, at least 12gb storage and 12gb vram
2. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. `source ~/.bashrc`
4. Install python: `uv python install 3.12`
5. Clone repo: `git clone https://github.com/Arrrlex/arena4-capstone.git`
6. Get api keys and put them in `.env` inside `arena4-capstone`
7. Install dependencies: `cd arena4-capstone && git switch use-uv && uv sync`
8. Install python and jupyter vscode extensions
