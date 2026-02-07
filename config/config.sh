sudo cp /home/pi/PiCameraArray/config/picamera-capture.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable picamera-capture.service
sudo systemctl start picamera-capture.service
