chmod +x main.py
sudo cp iusee.service /etc/systemd/system/
mkdir -p /home/pi/logs
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable iusee.service
sudo systemctl start iusee.service --now
sudo systemctl status iusee.service
journalctl -u iusee.service -e


