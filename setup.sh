!#/bin/bash

python3 -m pip install -U pip --break-system-packages
python3 -m pip install -e . --break-system-packages
# optional acceleration
python3 -m pip install -e ".[faiss]" --break-system-packages
