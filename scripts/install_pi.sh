#!/usr/bin/env bash
# One-time installer for Raspberry Pi 5 (Raspberry Pi OS Bookworm/Trixie, 64-bit).
# Usage:  cd ~/buddy && bash scripts/install_pi.sh
set -euo pipefail
cd "$(dirname "$0")/.."
APP_DIR="$(pwd)"
DATA="${BUDDY_HOME:-$HOME/buddy-data}"

echo "== 1/5 System packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libportaudio2 alsa-utils python3-gpiozero python3-lgpio curl bzip2

echo "== 2/5 Python environment (.venv, with access to system gpiozero)"
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "== 3/5 Speech models"
bash scripts/download_models.sh

echo "== 4/5 Config"
mkdir -p "$DATA"
if [ ! -f "$DATA/buddy.toml" ]; then
  cp buddy.example.toml "$DATA/buddy.toml"
  echo "   created $DATA/buddy.toml (edit child age, volume, etc.)"
fi

echo "== 5/5 Checks"
python -m buddy doctor || true

cat <<EOF

Next steps:
  1) Pre-render Buddy's voice (one time; Kokoro voice takes about 2-3 hours on a Pi 5,
     the faster Piper voice about 15 minutes - see docs/SETUP.md):
        cd $APP_DIR && . .venv/bin/activate && python -m buddy build-content
  2) Try it:        python -m buddy run
  3) Start on boot: sudo cp scripts/buddy.service /etc/systemd/system/ && \\
                    sudo sed -i "s#@USER@#$USER#g; s#@APP@#$APP_DIR#g" /etc/systemd/system/buddy.service && \\
                    sudo systemctl enable --now buddy
EOF
