#include "MKS_SERVO42.h"


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
  return float(esp_timer_get_time())*1000000.0;
}

float currentLineLen() {
  lenRecord[recordIdx][0] = float(millis())*1000.0;
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
  delay(1000);
  Serial.println();
  Serial.println("alive\n\n\n");

  Serial1.begin(38400, SERIAL_8N1, RXPIN, TXPIN);
  delay(100);
  stepper.initialize(&Serial1);

  stepper.ping(STID);
  delay(400);

  zeroAngle = float(stepper.getMotorShaftAngle(STID));
  Serial.printf("zeroAngle = %f\n", zeroAngle);
  delay(1);

  // populate a list with desired line lengths as if we had received it from the control center process
  float tt = float(millis())*0.001 + 0.1;
  for (int i=0; i<DATA_LEN; i++) {
    desiredLine[i][0] = tt;
    desiredLine[i][1] = waves(tt, frequency1, amplitude1, frequency2, amplitude2, phaseShift1, phaseShift2) + 3;
    tt += float(RUN_DELAY_MS) / 1000;
  }
}

void loop() {

  

  float t = millis()*0.001;
  // Find the earliest entry in desiredLine that is still in the future.
  while (desiredLine[lastIdx][0] <= t) {
    lastIdx++;
  }
  if (lastIdx >= DATA_LEN) {
    slowStop();
  }
  float targetLen = desiredLine[lastIdx][1];
  float currentLen = currentLineLen();
  float position_err = targetLen - currentLen;
  // What would the speed be between the two datapoints tha straddle the present?
  // Positive values mean line is lengthening
  float targetSpeed = ((desiredLine[lastIdx][1] - desiredLine[lastIdx-1][1]) 
    / (desiredLine[lastIdx][0] - desiredLine[lastIdx-1][0])); // in meters per second
  float currentSpeed = speed * speed1_revs * meters_per_rev * (dir==0 ? 1 : -1);
  float speed_err = targetSpeed - currentSpeed;
  // If our positional error was zero, we could go exactly that speed.
  // if our position was behind the targetLen (line is lengthening, and we are shorter than targetLen),
  // (or line is shortening and we are longer than target len) then we need to go faster than targetSpeed to catch up
  // ideally we want to catch up in one step, but we have max acceleration constraints.
  float aimSpeed = targetSpeed + position_err * PE_TERM;
  // Serial.printf("aimSpeed = %f, currentSpeed = %f m/s, targetSpeed = %f m/s, p-term = %f m/s\n", aimSpeed, currentSpeed, targetSpeed, position_err * PE_TERM);

  // light on when acceleration too high
  if (abs(position_err) > 0.3) {
    digitalWrite(FIREBEETLE2_LED, HIGH);
  } else {
    digitalWrite(FIREBEETLE2_LED, LOW);
  }

  // mStep = 16
  // Vrpm = (speed × 30000)/(mStep × 200) for a 1.8°motor
  // Speed can only go to 127 because we can only use the 7 lsb to represent it.
  // if you need to go faster, set the microsteps lower.
  speed = min(abs(aimSpeed / meters_per_rev / speed1_revs), 127.0f);
  dir = aimSpeed > 0 ? 0 : 1;
  Serial.printf("position_err = %f, speed = %u\n", position_err, speed);
  stepper.runMotorConstantSpeed(STID, dir, speed);
  delay(RUN_DELAY_MS);
}
