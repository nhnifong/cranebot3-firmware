#include "MKS_SERVO42.h"
#include <WiFi.h>
#include "esp_camera.h"
#include "DFRobot_AXP313A.h"

// ===================
// Camera Pins
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

#define FIREBEETLE2_LED 21
#define RXPIN 44
#define TXPIN 43
#define STID 0
#define ANGLE_RESOLUTION 65535
#define SPOOL_DIAMETER_MM 23.0f
#define DATA_LEN 1000
#define PE_TERM 1.5f
#define DEFAULT_MICROSTEPS 16
#define RUN_DELAY_MS 30

// Power manager for camera
DFRobot_AXP313A axp;

// ===========================
// Enter your WiFi credentials
// ===========================
const char* ssid = "ATTEdnN5S2";
const char* password = "gwc549+9e7e4";

void startCameraServer();

// A speed of 1 is this many revs/sec
const float speed1_revs = 30000.0/(DEFAULT_MICROSTEPS * 200)/60;
const float meters_per_rev = SPOOL_DIAMETER_MM * PI * 0.001;

MKS_SERVO42 stepper;


float zeroAngle=0;
float currentAngle=0;
// Dir 0 is when line is being unspooled from the top. (top moves away from the wall)
// In the unspooling direction, the angle reported by getMotorShaftAngle is decreasing.
byte dir = 0;
uint8_t speed = 0;
// Meters of line that were spooled out when zeroAngle was set.
float lineAtStart = 2;

// Array of desired line lengths
// col 0: time in seconds
// col 1: length in meters
float desiredLine[DATA_LEN][2];
// The earliest index of this array that is known to be in the future as of the last time we checked.
// Starts at 1 because we do a look back.
uint16_t lastIdx = 1;

// array to store actual line lengths for analysis
float lenRecord[DATA_LEN][2];
uint16_t recordIdx = 0;

float frequency1 = 0.041;
float amplitude1 = 0.3;
float frequency2 = 0.121;
float amplitude2 = 0.04;
float phaseShift1 = PI/7; //Example phase shift
float phaseShift2 = PI/3; //Example phase shift

float waves(float time, float freq1, float amp1, float freq2, float amp2, float phase1 = 0.0, float phase2 = 0.0) {
  // time: The current time (in whatever units you're using - seconds, milliseconds, etc.)
  // freq1: Frequency of the first wave (e.g., in Hz)
  // amp1: Amplitude of the first wave
  // freq2: Frequency of the second wave
  // amp2: Amplitude of the second wave
  // phase1: Phase shift of the first wave (in radians) - default 0
  // phase2: Phase shift of the second wave (in radians) - default 0

  float wave1 = amp1 * sin(2 * PI * freq1 * time + phase1);
  float wave2 = amp2 * cos(2 * PI * freq2 * time + phase2); // Using cosine for the second wave to show variety

  return wave1 + wave2;
}

void slowStop() {
  while (speed > 0) {
    speed--;
    stepper.runMotorConstantSpeed(STID, dir, speed);
  }
  stepper.stopMotor(STID);
  Serial.println("stopped motor");
  while(true) {
    delay(10);
  }
}

float now() {
  return float(esp_timer_get_time())/1000000;
}

float currentLineLen() {
  lenRecord[recordIdx][0] = now();
  float len = -1 * meters_per_rev * (float(stepper.getMotorShaftAngle(STID)) - zeroAngle) / ANGLE_RESOLUTION + lineAtStart;
  lenRecord[recordIdx][1] = len;
  return len;
}

uint8_t sign(float u) {
  return u>0;
}

void setup() {
  pinMode(FIREBEETLE2_LED, OUTPUT);

  Serial.begin(115200);
  Serial.setDebugOutput(true);
  delay(400);
  Serial.println();
  Serial.println("alive\n\n\n");

  // Set up camera power
  while(axp.begin() != 0){
    Serial.println("AXP313A init error");
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
  // s->set_vflip(s, 1); // flip it back
  // s->set_brightness(s, 1); // up the brightness just a bit
  // s->set_saturation(s, -2); // lower the saturation

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  Serial.print("Establishing WiFi Connection");
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

  // Set up uart serial port to MKS_SERVO42C
  Serial1.begin(38400, SERIAL_8N1, RXPIN, TXPIN);
  delay(100);
  stepper.initialize(&Serial1);

  stepper.ping(STID);
  delay(100);

  zeroAngle = float(stepper.getMotorShaftAngle(STID));
  Serial.printf("zeroAngle = %f\n", zeroAngle);
  delay(1);

  // populate a list with desired line lengths as if we had received it from the control center process
  float tt = now() + 0.1;
  for (int i=0; i<DATA_LEN; i++) {
    desiredLine[i][0] = tt;
    desiredLine[i][1] = waves(tt, frequency1, amplitude1, frequency2, amplitude2, phaseShift1, phaseShift2) + 3;
    tt += float(RUN_DELAY_MS) / 1000;
  }


}

void loop() {
  // Everything for the camera is done in another task by the web server during the delays

  // float t = now();
  // // Find the earliest entry in desiredLine that is still in the future.
  // while (desiredLine[lastIdx][0] <= t) {
  //   lastIdx++;
  // }
  // if (lastIdx >= DATA_LEN) {
  //   slowStop();
  // }
  // float targetLen = desiredLine[lastIdx][1];
  // float currentLen = currentLineLen();
  // float position_err = targetLen - currentLen;
  // // What would the speed be between the two datapoints tha straddle the present?
  // // Positive values mean line is lengthening
  // float targetSpeed = ((desiredLine[lastIdx][1] - desiredLine[lastIdx-1][1]) 
  //   / (desiredLine[lastIdx][0] - desiredLine[lastIdx-1][0])); // in meters per second
  // float currentSpeed = speed * speed1_revs * meters_per_rev * (dir==0 ? 1 : -1);
  // float speed_err = targetSpeed - currentSpeed;
  // // If our positional error was zero, we could go exactly that speed.
  // // if our position was behind the targetLen (line is lengthening, and we are shorter than targetLen),
  // // (or line is shortening and we are longer than target len) then we need to go faster than targetSpeed to catch up
  // // ideally we want to catch up in one step, but we have max acceleration constraints.
  // float aimSpeed = targetSpeed + position_err * PE_TERM;
  // // Serial.printf("aimSpeed = %f, currentSpeed = %f m/s, targetSpeed = %f m/s, p-term = %f m/s\n", aimSpeed, currentSpeed, targetSpeed, position_err * PE_TERM);

  // // light on when acceleration too high
  // if (abs(position_err) > 0.3) {
  //   digitalWrite(FIREBEETLE2_LED, HIGH);
  // } else {
  //   digitalWrite(FIREBEETLE2_LED, LOW);
  // }

  // // mStep = 16
  // // Vrpm = (speed × 30000)/(mStep × 200) for a 1.8°motor
  // // Speed can only go to 127 because we can only use the 7 lsb to represent it.
  // // if you need to go faster, set the microsteps lower.
  // speed = min(abs(aimSpeed / meters_per_rev / speed1_revs), 127.0f);
  // dir = aimSpeed > 0 ? 0 : 1;
  // Serial.printf("position_err = %f, speed = %u\n", position_err, speed);
  // stepper.runMotorConstantSpeed(STID, dir, speed);
  delay(RUN_DELAY_MS);
}
