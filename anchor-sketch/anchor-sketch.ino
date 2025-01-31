#include "MKS_SERVO42.h"


#define FIREBEETLE2_LED 21
#define RXPIN 44
#define TXPIN 43

byte const stepperId = 0;
uint8_t const speed = 20;
uint32_t const numberOfPulses = 10000;

long current_pos = 0;
byte dir = 0;
int count=0;

MKS_SERVO42 stepper;

void setup() {
  pinMode(FIREBEETLE2_LED, OUTPUT);

  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();
  Serial.println("alive");

  Serial1.begin(38400, SERIAL_8N1, RXPIN, TXPIN);
  stepper.initialize(&Serial1);
  stepper.ping(stepperId);
  Serial.println("setaccel");
  bool res = stepper.setAcceleration(stepperId, 8);
  Serial.println(res);
}

void loop() {
  // Serial.println(stepper.getCurrentPosition(stepperId));
  if(dir==0) {
    dir = 1;
    stepper.setTargetPosition(stepperId, dir, 50, numberOfPulses);
    digitalWrite(FIREBEETLE2_LED, HIGH);
  } else {
    dir = 0;
    stepper.setTargetPosition(stepperId, dir, 10, numberOfPulses);
    digitalWrite(FIREBEETLE2_LED, LOW);
  }
	delay(1000);
  count++;
  if (count==10){
    delay(100000);
  }
}
