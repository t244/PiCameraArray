# Update tye system
sudo apt update && sudo apt full-upgrade -y

# Install libcamera apps
sudo apt install -y libcamera-apps python3-picamera2

# Enable Wayland service
sudo systemctl enable wayvnc.service
sudo systemctl start wayvnc.service

# Set up VNC headless display
echo "" | sudo tee -a /boot/firmware/config.txt
echo "# VNC headless display settings" | sudo tee -a /boot/firmware/config.txt
echo "hdmi_force_hotplug=1" | sudo tee -a /boot/firmware/config.txt
echo "hdmi_group=2" | sudo tee -a /boot/firmware/config.txt
echo "hdmi_mode=87" | sudo tee -a /boot/firmware/config.txt

# Set up the camera overlay for IMX296
echo "" | sudo tee -a /boot/firmware/config.txt
echo "# Camera overlay for IMX296" | sudo tee -a /boot/firmware/config.txt
echo "dtoverlay=imx296" | sudo tee -a /boot/firmware/config.txt

# Replace camera_auto_detect=1 with 0
sudo sed -i 's/camera_auto_detect=1/camera_auto_detect=0/' /boot/firmware/config.txt

# Reboot to apply changes
sudo reboot
