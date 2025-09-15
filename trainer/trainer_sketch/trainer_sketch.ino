// Sketch to run trainer controller. reads buttons and trigger and transmits over BT
// Based on the original BLE_uart example.

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// UUIDs remain the same from the original UART example for simplicity.
#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E" // UART service UUID
#define CHARACTERISTIC_UUID_RX "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

#define TRIG_PIN A3
#define BUT_A_PIN D6
#define BUT_B_PIN D9
#define BUT_C_PIN D7
#define RELAXED 1500
#define PULLED 1050

int sensorValue = 0;
float triggerFrac  = 0;

BLEServer *pServer = NULL;
BLECharacteristic *pTxCharacteristic;
bool deviceConnected = false;

// --- BLE Connection Callbacks ---
// This class handles events when a device connects or disconnects.
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      Serial.println("Bluetooth connected");
      deviceConnected = true;
    };

    void onDisconnect(BLEServer* pServer) {
      Serial.println("Bluetooth disconnected");
      deviceConnected = false;
      // Restart advertising so a new connection can be made.
      pServer->getAdvertising()->start();
    }
};

// --- BLE Data Received Callback ---
// This class handles data received from the connected device.
// For a simple controller, this might not be used, but it's here if you need it.
class MyCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
      String rxValue = pCharacteristic->getValue().c_str();

      if (rxValue.length() > 0) {
        Serial.print("Received Value: ");
        Serial.println(rxValue);
      }
    }
};

void setup(){
   pinMode(TRIG_PIN, INPUT);
   pinMode(BUT_A_PIN, INPUT_PULLUP);
   pinMode(BUT_B_PIN, INPUT_PULLUP);
   pinMode(BUT_C_PIN, INPUT_PULLUP);
   Serial.begin(11520);   //Initialize serial port


  BLEDevice::init("Stringman Training Controller"); // name of BLE device
  
  // This enables "Just Works" pairing, which doesn't require a PIN or passkey.
  BLESecurity *pSecurity = new BLESecurity();
  pSecurity->setAuthenticationMode(ESP_LE_AUTH_NO_BOND);

  // Create the BLE Server
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());
  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Create a BLE Characteristic for sending data (Transmit)
  pTxCharacteristic = pService->createCharacteristic(
                    CHARACTERISTIC_UUID_TX,
                    BLECharacteristic::PROPERTY_NOTIFY
                  );
  pTxCharacteristic->addDescriptor(new BLE2902());

  // Create a BLE Characteristic for receiving data (Receive)
  BLECharacteristic * pRxCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID_RX,
                      BLECharacteristic::PROPERTY_WRITE
                    );
  pRxCharacteristic->setCallbacks(new MyCallbacks());

  // Start the service and start advertising
  pService->start();
  pServer->getAdvertising()->start();
  Serial.println("Waiting for a client connection...");
}

void loop(){

  // Read trigger and add smoothing
  sensorValue = analogRead(TRIG_PIN);
  float rawFrac = constrain(
    static_cast<float>(RELAXED - sensorValue) / (RELAXED - PULLED),
    0, 1);
  triggerFrac = triggerFrac*0.4 + rawFrac*0.6;
  // Serial.printf("Trigger : %0.2f%%\n", triggerFrac*100);  //Print the read sensor value

  // Read buttons and invert
  int buttonStateA = !digitalRead(BUT_A_PIN);
  int buttonStateB = !digitalRead(BUT_B_PIN);
  int buttonStateC = !digitalRead(BUT_C_PIN);
  // Serial.printf("buttons : %d,%d,%d\n", buttonStateA, buttonStateB, buttonStateC);  //Print the read sensor value

  // if any client is connected, send them the button and trigger states
  if (deviceConnected) {

    // Create a comma-separated string with the input states.
    // Example: "1,0,1,0.0224"
    String dataToSend = String(buttonStateA) + "," + String(buttonStateB) + "," + String(buttonStateC) + "," + String(triggerFrac);

    // Set the value of the characteristic and send a notification.
    pTxCharacteristic->setValue(dataToSend.c_str());
    pTxCharacteristic->notify();
  }

  delay(50);
}
