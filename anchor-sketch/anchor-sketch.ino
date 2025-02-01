#include "MKS_SERVO42.h"


#define FIREBEETLE2_LED 21
#define RXPIN 44
#define TXPIN 43

byte const stepperId = 0;
MKS_SERVO42 stepper;


double zeropos=0;
double angle=0;

float frequency1 = 1.1;
float amplitude1 = 1;
float frequency2 = 0.312;
float amplitude2 = 2.0;
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

void setup() {
  pinMode(FIREBEETLE2_LED, OUTPUT);

  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();
  Serial.println("alive");

  Serial1.begin(38400, SERIAL_8N1, RXPIN, TXPIN);
  delay(100);
  stepper.initialize(&Serial1);

  stepper.ping(stepperId);
  delay(400);

  zeropos = double(stepper.getMotorShaftAngle(stepperId));
  Serial.printf("zeropos = %f\n", zeropos);
  delay(1);
}

void loop() {

  

  float t = millis()/1000.0;
  double desired_angle = waves(t, frequency1, amplitude1, frequency2, amplitude2, phaseShift1, phaseShift2) * 100000;
  angle = double(stepper.getMotorShaftAngle(stepperId)) - zeropos;
  double angle_error = angle - desired_angle;
  // if the angle error is positive, you want direction 0, if it's negative, you want direction 1
  byte dir = 0;
  if (angle_error < 0) {
    dir = 1;
  }
  uint8_t speed = min(abs(angle_error)/1000, 100.0);
  Serial.printf("des = %f, angle = %f, err = %f, speed = %u\n", desired_angle, angle, angle_error, speed);
  stepper.runMotorConstantSpeed(stepperId, dir, speed);
  delay(20);


  // angle = stepper.getMotorShaftAngle(stepperId);
  // angle_error = angle - desired_angle;
  // Serial.println(angle_error);
  
  // stepper.stopMotor(stepperId);
  // delay(100000);

  // if (flag && t > 8000) {
  //   Serial.printf("move back %ul\n", t);
  //   // printpos();
  //   // stepper.setTargetPosition(stepperId, 0, 20, 4*3200);
  //   stepper.runMotorConstantSpeed(stepperId, 0, 18);
  //   flag = false;
  // }

  // // this seems to return tenths of a revolution as well.
  // if (millis() - start < 16000) {
  //   printpos();
  //   delay(50);
  // } else {
  //   printpos();
  //   stepper.stopMotor(stepperId);
  //   delay(100000);
  // }

  // if (count==10){
  //   delay(100000);
  // }
}
