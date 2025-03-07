if [ ! -d .git ]; then
    git clone https://github.com/Arrrlex/arena4-capstone.git
    cd arena4-capstone
fi


apt update
apt install -y curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc && uv python install 3.12

source $HOME/.local/bin/env
uv sync