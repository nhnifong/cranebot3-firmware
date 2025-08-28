/*
    BLE Game Controller for ESP32-S3 (Firebeetle 2)
    
    This sketch transforms the ESP32-S3 into a Bluetooth game controller.
    It reads the state of 3 buttons and 1 analog trigger and sends the data
    to a connected central device (e.g., a Raspberry Pi) via BLE notifications.

    Based on the original BLE_uart example.

    How to flash this
    https://wiki.dfrobot.com/FireBeetle_Board_ESP32_E_SKU_DFR0654
*/
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// --- Pin Definitions ---
// Define the GPIO pins for your buttons and analog trigger.
// Connect one side of the buttons to these pins and the other to GND.
const int BUTTON1_PIN = D2;
const int BUTTON2_PIN = D3;
const int BUTTON3_PIN = D4;

// Connect your analog trigger (e.g., potentiometer's middle pin) to this pin.
const int ANALOG_TRIGGER_PIN = A0;


// --- BLE Setup ---
BLEServer *pServer = NULL;
BLECharacteristic *pTxCharacteristic;
bool deviceConnected = false;

// UUIDs remain the same from the original UART example for simplicity.
#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E" // UART service UUID
#define CHARACTERISTIC_UUID_RX "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"


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
      String rxValue = pCharacteristic->getValue();

      if (rxValue.length() > 0) {
        Serial.print("Received Value: ");
        Serial.println(rxValue);
      }
    }
};


// --- Main Setup Function ---
void setup() {
  Serial.begin(115200);
  Serial.println("Starting BLE Game Controller...");

  // --- Initialize GPIO Pins ---
  // Set button pins as inputs with an internal pull-up resistor.
  // This means the pin will be HIGH when the button is not pressed and LOW when pressed.
  pinMode(BUTTON1_PIN, INPUT_PULLUP);
  pinMode(BUTTON2_PIN, INPUT_PULLUP);
  pinMode(BUTTON3_PIN, INPUT_PULLUP);

  // The analog pin doesn't need a pinMode declaration for analogRead.

  // --- Initialize Bluetooth ---
  BLEDevice::init("Firebeetle Game Controller"); // Set the name of your BLE device

  // Create the BLE Server
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create the BLE Service
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

  // Start the service
  pService->start();

  // Start advertising
  pServer->getAdvertising()->start();
  Serial.println("Waiting for a client connection...");
}


// --- Main Loop Function ---
void loop() {
  // Check if a device is connected
  if (deviceConnected) {
    // --- Read Inputs ---
    // digitalRead will be LOW when a button is pressed due to INPUT_PULLUP.
    // We invert the logic with '!' so that '1' means pressed and '0' means not pressed.
    int button1State = !digitalRead(BUTTON1_PIN);
    int button2State = !digitalRead(BUTTON2_PIN);
    int button3State = !digitalRead(BUTTON3_PIN);
    
    // Read the value from the analog pin. This will be a value from 0 to 4095 on the ESP32-S3.
    int analogValue = analogRead(ANALOG_TRIGGER_PIN);

    // --- Format Data ---
    // Create a comma-separated string with the input states.
    // Example: "1,0,1,2048"
    String dataToSend = String(button1State) + "," + String(button2State) + "," + String(button3State) + "," + String(analogValue);

    // --- Send Data ---
    // Set the value of the characteristic and send a notification.
    pTxCharacteristic->setValue(dataToSend.c_str());
    pTxCharacteristic->notify();
    
    // Print to serial monitor for debugging
    Serial.println("Sent: " + dataToSend);

    // Delay to control the rate of transmission (e.g., 20 times per second).
    delay(50); 
  }
}