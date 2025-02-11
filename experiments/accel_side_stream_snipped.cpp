#include <ArduinoJson.h> // Make sure you have this library installed

static esp_err_t stream_handler(httpd_req_t *req) {
    // ... (Existing code for camera setup and headers)

    while (true) {
        // ... (Existing code for capturing and encoding the image)

        if (res == ESP_OK) {
            // Get acceleration data (from your previous code)
            DFRobot_BNO055::sAxisData_t accelRaw = bno.getAxisRaw(DFRobot_BNO055::eAxisAcc);
            DFRobot_BNO055::sQuaAnalog_t quat = bno.getQua();
            // ... (Convert raw acceleration and quaternion to global acceleration)

            // Create a JSON object for the acceleration data
            StaticJsonDocument accelJson; // Adjust size as needed
            accelJson["x"] = linearAccelGlobal.x;
            accelJson["y"] = linearAccelGlobal.y;
            accelJson["z"] = linearAccelGlobal.z;
            // Add timestamp if needed
            accelJson["timestamp"] = _timestamp.tv_sec + (_timestamp.tv_usec / 1000000.0);
            char accelBuffer[256]; // Buffer to hold the JSON string
            serializeJson(accelJson, accelBuffer);

            // Send the boundary and image header
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
            if (res != ESP_OK) break;

            size_t hlen = snprintf((char *)part_buf, 128, _STREAM_PART, _jpg_buf_len, _timestamp.tv_sec, _timestamp.tv_usec);
            res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
            if (res != ESP_OK) break;

            // Send the image data
            res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
            if (res != ESP_OK) break;

            // Send the acceleration data as a separate chunk after image
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
            if (res != ESP_OK) break;

            size_t accel_hlen = snprintf((char *)part_buf, 128, _STREAM_PART_ACCEL, strlen(accelBuffer), _timestamp.tv_sec, _timestamp.tv_usec);
            res = httpd_resp_send_chunk(req, (const char *)part_buf, accel_hlen);
            if (res != ESP_OK) break;

            res = httpd_resp_send_chunk(req, accelBuffer, strlen(accelBuffer));
            if (res != ESP_OK) break;
        }

        // ... (Existing code for returning the frame and error handling)
    }

    // ... (Existing code for LED and return)
}