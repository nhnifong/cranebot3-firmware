#include "esp_camera.h"
#include <WiFi.h>
#include "DFRobot_AXP313A.h"
#include "ESP32_New_ISR_Servo.h"

// WARNING!!! PSRAM IC required for UXGA resolution and high JPEG quality
// Must select the following settings from tools menu before burning
// 
// USB CDC on boot "Enabled"
// Flash Size: "16MB (128Mb)"
// Partition Scheme "16M Flash (3MB APP/9.9MB FATFS)"
// PSRAM "OPI PSRAM"

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

// Servo constants
#define MIN_MICROS      500  //544
#define MAX_MICROS      2500

#define NUM_SERVOS    2

// Pressure resistor
#define PRESSURE_PIN 11 //A5

int sensorValue = 0;  // variable to store the value coming from the voltage divider.

typedef struct
{
	int     servoIndex;
	uint8_t servoPin;
} ISR_servo_t;

ISR_servo_t ISR_servo[NUM_SERVOS] =
{
	{ -1, 12 }, // gripper blue D12
  { -1, 13 }, // winch green  D11
};

void startCameraServer();

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();
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
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG; // for streaming
  //config.pixel_format = PIXFORMAT_RGB565; // for face detection/recognition
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  //                      for larger pre-allocated frame buffer.
  if(config.pixel_format == PIXFORMAT_JPEG){
    if(psramFound()){
      config.jpeg_quality = 10;
      config.fb_count = 2;
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      // Limit the frame size when PSRAM is not available
      config.frame_size = FRAMESIZE_SVGA;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    // Best option for face detection/recognition
    config.frame_size = FRAMESIZE_240X240;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

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
  // drop down frame size for higher initial frame rate
  if(config.pixel_format == PIXFORMAT_JPEG){
    s->set_framesize(s, FRAMESIZE_QVGA);
  }

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

#if defined(CAMERA_MODEL_ESP32S3_EYE)
  s->set_vflip(s, 1);
#endif

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  startCameraServer();

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");


  Serial.println("Setup servos...");
	ESP32_ISR_Servos.useTimer(USE_ESP32_TIMER_NO);
  for (int index = 0; index < NUM_SERVOS; index++)
	{
		ISR_servo[index].servoIndex = ESP32_ISR_Servos.setupServo(ISR_servo[index].servoPin, MIN_MICROS, MAX_MICROS);

		if (ISR_servo[index].servoIndex != -1)
		{
			Serial.print(F("Setup OK Servo index = "));
			Serial.print(ISR_servo[index].servoIndex);
			Serial.print(F(" pin = "));
			Serial.println(ISR_servo[index].servoPin);
		}
		else
		{
			Serial.print(F("Setup Failed Servo index = "));
			Serial.println(ISR_servo[index].servoIndex);
		}
	}
}

void loop() {
  // Everything for the camera is done in another task by the web server during the delays

  // ESP32_ISR_Servos.setPosition(ISR_servo[0].servoIndex, 0);
  // ESP32_ISR_Servos.setPosition(ISR_servo[1].servoIndex, 0);
	delay(1000);
  // ESP32_ISR_Servos.setPosition(ISR_servo[0].servoIndex, 180);
  // ESP32_ISR_Servos.setPosition(ISR_servo[1].servoIndex, 180);
	delay(1000);

  sensorValue = analogRead(PRESSURE_PIN);
	Serial.println(sensorValue);
}