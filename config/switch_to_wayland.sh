# Stop RealVNC first
sudo systemctl stop vncserver-x11-serviced.service
sudo systemctl disable vncserver-x11-serviced.service

# Enable Wayland service
sudo systemctl enable wayvnc.service
sudo systemctl start wayvnc.service
