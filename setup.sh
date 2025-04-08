apt update
apt install -y curl
apt install -y git

if [ ! -d .git ]; then
    git clone https://github.com/Arrrlex/arena4-capstone.git
    cd arena4-capstone
fi



curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

source $HOME/.local/bin/env
UV_PYTHON_INSTALL_DIR=./uv_python/ uv sync