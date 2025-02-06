#include "esp_camera.h"
#include <WiFi.h>
#include "DFRobot_AXP313A.h"
#include "ESP32_New_ISR_Servo.h"
#include "DFRobot_BNO055.h"
#include "Wire.h"
#include "other_record_types.h"

// WARNING!!! PSRAM IC required for UXGA resolution and high JPEG quality
// Must select the following settings from tools menu before burning
// 
// USB CDC on boot "Enabled"
// Flash Size: "16MB (128Mb)"
// Partition Scheme "16M Flash (3MB APP/9.9MB FATFS)"
// PSRAM "OPI PSRAM"

#define FIREBEETLE2_LED 21

#define USE_ESP32_TIMER_NO          3

// ===================
// Select camera model
// ===================
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     45
#define SIOD_GPIO_NUM     1
#define SIOC_GPIO_NUM     2

#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       46
#define Y7_GPIO_NUM       8
#define Y6_GPIO_NUM       7
#define Y5_GPIO_NUM       4
#define Y4_GPIO_NUM       41
#define Y3_GPIO_NUM       40
#define Y2_GPIO_NUM       39
#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     42
#define PCLK_GPIO_NUM     5

DFRobot_AXP313A axp;

// ===========================
// Enter your WiFi credentials
// ===========================
const char* ssid = "ATTEdnN5S2";
const char* password = "gwc549+9e7e4";

// queue to store actual line lengths for sending back over http
QueueHandle_t queue;
int queueSize = 40;

// Servo constants
#define MIN_MICROS      500  //544
#define MAX_MICROS      2500

#define WINCH_PIN       12
#define GRIP_PIN        13

int winch = -1;
int grip = -1;

// Pressure resistor
#define PRESSURE_PIN 11 //A5

int sensorValue = 0;  // variable to store the value coming from the voltage divider.
int time_to_open = esp_timer_get_time(); // when to open the hand

typedef DFRobot_BNO055_IIC    BNO;
BNO   bno(&Wire, 0x28);    // input TwoWire interface and IIC address for IMU

void startCameraServer(QueueHandle_t q);

// show last sensor operate status
void printLastOperateStatus(BNO::eStatus_t eStatus)
{
  switch(eStatus) {
  case BNO::eStatusOK:    Serial.println("everything ok"); break;
  case BNO::eStatusErr:   Serial.println("unknow error"); break;
  case BNO::eStatusErrDeviceNotDetect:    Serial.println("device not detected"); break;
  case BNO::eStatusErrDeviceReadyTimeOut: Serial.println("device ready time out"); break;
  case BNO::eStatusErrDeviceStatus:       Serial.println("device internal status error"); break;
  default: Serial.println("unknown status"); break;
  }
}

// Function to perform quaternion multiplication
BNO::sQuaAnalog_t quatMultiply(BNO::sQuaAnalog_t q1, BNO::sQuaAnalog_t q2) {
  BNO::sQuaAnalog_t result;
  result.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
  result.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
  result.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
  result.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
  return result;
}

// Function to rotate a vector by a quaternion
BNO::sAxisAnalog_t rotateVector(BNO::sAxisAnalog_t v, BNO::sQuaAnalog_t q) {
  BNO::sQuaAnalog_t v_quat = {0, v.x, v.y, v.z}; // Convert vector to quaternion
  BNO::sQuaAnalog_t q_conj = {q.w, -q.x, -q.y, -q.z}; // Quaternion conjugate
  BNO::sQuaAnalog_t rotated_quat = quatMultiply(quatMultiply(q, v_quat), q_conj);
  BNO::sAxisAnalog_t result;
  result.x = rotated_quat.x;
  result.y = rotated_quat.y;
  result.z = rotated_quat.z;
  return result;
}

void setup() {
  pinMode(FIREBEETLE2_LED, OUTPUT);

  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  queue = xQueueCreate( queueSize, sizeof( line_record_t ) );
  if(queue == NULL){
    Serial.println("Error creating the queue");
  }

  Serial.println("Setup IMU...");
  bno.reset();
  while(bno.begin() != BNO::eStatusOK) {
    Serial.println("IMU init failed");
    printLastOperateStatus(bno.lastOperateStatus);
    delay(500);
  }
  Serial.println("IMU init  success");

  // Set up camera power
  while(axp.begin() != 0){
    Serial.println("init error");
    delay(1000);
  }
  axp.enableCameraPower(axp.eOV2640);//Enable the power for camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  // Setting this to -1 means "use an already initialzied i2c driver" which is why we init the IMU first.
  // see https://github.com/espressif/esp32-camera/pull/413
  // This is the only way to get the camera and IMU to play nice on the same I2C bus
  config.pin_sscb_sda = -1; // SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG; // for streaming
  //config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10; // A lower number is a higher quality. range 0-63
  config.fb_count = 2; // framebuffer can be 2 on the ESP32-S3.

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1); // flip it back
    s->set_brightness(s, 1); // up the brightness just a bit
    s->set_saturation(s, -2); // lower the saturation
  }

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  Serial.print("Establishing WiFi Connection");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  startCameraServer(queue);
  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");

  Serial.println("Setup servos...");
	ESP32_ISR_Servos.useTimer(USE_ESP32_TIMER_NO);
  winch = ESP32_ISR_Servos.setupServo(WINCH_PIN, MIN_MICROS, MAX_MICROS);
  grip = ESP32_ISR_Servos.setupServo(GRIP_PIN, MIN_MICROS, MAX_MICROS);

  ESP32_ISR_Servos.setPosition(winch, 90);
  ESP32_ISR_Servos.setPosition(grip, 0);

  digitalWrite(FIREBEETLE2_LED, HIGH);
}

void loop() {
  // Everything for the camera is done in another task by the web server during the delays

  // read finger pressure
  sensorValue = analogRead(PRESSURE_PIN);
	Serial.print("Finger pressure: ");
	Serial.println(sensorValue);

  // Set winch to middle position (not moving)
  if (sensorValue > 1000 && sensorValue < 2000) {
    ESP32_ISR_Servos.setPosition(winch, 120);
  } else if (sensorValue > 2000) {
    ESP32_ISR_Servos.setPosition(winch, 60);
  } else {
    ESP32_ISR_Servos.setPosition(winch, 90);
  }

  // if (esp_timer_get_time() > time_to_open) {
  //   // Set grip to fully open
  //   ESP32_ISR_Servos.setPosition(grip, 0);
  //   if (sensorValue > 1000) {
  //     time_to_open = esp_timer_get_time() + 5000000; // close for 5 seconds
  //   }
  // } else {
  //   // Set grip to fully closed
  //   ESP32_ISR_Servos.setPosition(grip, 180);
  // }

  // Get linear acceleration. this should already have gravity subtracted from it.
  // BNO::sAxisData_t accelRaw = bno.getAxisRaw(DFRobot_BNO055::eAxisAcc);
  // BNO::sQuaAnalog_t quat = bno.getQua();

  // Rotate linear acceleration to global frame
  // BNO::sAxisAnalog_t linearAccelGlobal = rotateVector(linearAccelDevice, quat);

  // Print acceleration in global frame
  // Serial.print("X accel global:");
  // Serial.print(linearAccelGlobal.x, 3);
  // Serial.print(" ");
  // Serial.print("Y accel global:");
  // Serial.print(linearAccelGlobal.y, 3);
  // Serial.print(" ");
  // Serial.print("Z accel global:");
  // Serial.print(linearAccelGlobal.z, 3);
  // Serial.println(" ");

  // digitalWrite(FIREBEETLE2_LED, HIGH);
	// delay(500);
  // digitalWrite(FIREBEETLE2_LED, LOW);
	// delay(500);

  delay(30);
}