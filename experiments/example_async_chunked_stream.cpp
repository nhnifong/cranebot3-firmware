#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

AsyncWebServer server(80);

// ... (Camera and image encoding setup) ...

// Global variable to hold the response object (BAD PRACTICE, but for demonstration)
AsyncWebServerResponse *streamResponse;

void setup() {
    // ... (Serial, camera initialization) ...

    server.on("/stream", [](AsyncWebServerRequest *request){
        streamResponse = request->beginResponse(200, "image/jpeg"); // Or appropriate content type
        streamResponse->chunked();  // Enable chunked encoding
        request->send(streamResponse); // Send initial headers, don't wait for data!
    });

    server.begin();
}

void loop() {
    // ... (Other tasks) ...

    if (streamResponse) { // Check if a stream request is active
      // Encode image
      std::vector<uint8_t> frame_encoded;
      bool result = encodeFrame(frame_encoded);
      if (result) {
        streamResponse->writeChunk(frame_encoded.data(), frame_encoded.size()); // Send the frame chunk
      } else {
        streamResponse->client()->close(); // Close connection if encoding fails
        streamResponse = nullptr; // Reset response pointer
      }
    }

    delay(30); // Or other timing mechanism for frame rate
}