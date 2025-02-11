Code for the firebeetle2 controller versions of the anchor and gripper (abandoned)


required libraries for firebeetle sketches:

 - AXP313A - https://github.com/cdjq/DFRobot_AXP313A
 - ESP32_New_ISR_Servo - https://github.com/khoih-prog/ESP32_New_ISR_Servo
 - DFRobot_BNO055 - https://github.com/DFRobot/DFRobot_BNO055
 - DFRobot_BMP280 - https://github.com/DFRobot/DFRobot_BMP280


Follow the Arduino IDE configuration steps for the DF Robot Firebeetle 2

  https://wiki.dfrobot.com/SKU_DFR0975_FireBeetle_2_Board_ESP32_S3#target_6

OR

  In case that link rots, install the esp32 board using this manager url https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
  
  Set the sketch board type to "DF Robot Firebeetle 2 ESP32-S3"
  
  Select the following settings from tools menu before burning
	
 - USB CDC on boot "Enabled"
 - Flash Size: "16MB (128Mb)"
 - Partition Scheme "16M Flash (3MB APP/9.9MB FATFS)"
 - PSRAM "OPI PSRAM"