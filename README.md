## Setup

1. Install dependencies into venv from pyproject.toml/requirements.txt
2. Start venv
3. Run `server.py` to start aggregator
4. Open a new terminal and start the first client by running `client.py --partition 0`
5. Open a new terminal and start the second client by running `client.py --partition 1`