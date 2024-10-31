git clone -y https://github.com/arena4-ai/arena4-capstone.git
curl -LsSf https://astral.sh/uv/install.sh | sh
echo "source ~/.cargo/env" >> ~/.bashrc
source ~/.bashrc
cd arena4-capstone
uv python install 3.12
git switch use-uv
uv sync

echo "Don't forget to add env variables to .env"