#include "esp_http_server.h"
#include "esp_camera.h"
#include "esp32-hal-log.h"
#include "line_record_type.h"
#include <ArduinoJson.h> 

#define PART_BOUNDARY "123456789000000000000987654321"

static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *_STREAM_IMAGE_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %f\r\n\r\n";
static const char *_STREAM_JSON_PART = "Content-Type: application/json\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t stream_httpd = NULL;
QueueHandle_t lenQueue;

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  struct timeval _timestamp;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t *_jpg_buf = NULL;
  char *part_buf[128];
  char json_buf[1024];

  httpd_ws_frame_t ws_pkt;

  static int64_t last_frame = 0;
  if (!last_frame) {
    last_frame = esp_timer_get_time();
  }

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) {
    return res;
  }

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "X-Framerate", "60");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      log_e("Camera capture failed");
      res = ESP_FAIL;
    } else {
      _timestamp.tv_sec = fb->timestamp.tv_sec;
      _timestamp.tv_usec = fb->timestamp.tv_usec;
      // we shouldn't hit this case, we're using the right pix format to begin with. it's compressed on the camera.
      if (fb->format != PIXFORMAT_JPEG) {
        bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
        esp_camera_fb_return(fb);
        fb = NULL;
        if (!jpeg_converted) {
          log_e("JPEG compression failed");
          res = ESP_FAIL;
        }
      } else {
        _jpg_buf_len = fb->len;
        _jpg_buf = fb->buf;
      }
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }
    if (res == ESP_OK) {
      size_t hlen = snprintf((char *)part_buf, 128, _STREAM_IMAGE_PART, _jpg_buf_len, float(_timestamp.tv_sec)+_timestamp.tv_usec*0.000001);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if (fb) {
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if (_jpg_buf) {
      free(_jpg_buf);
      _jpg_buf = NULL;
    }
    if (res != ESP_OK) {
      log_e("Send frame failed");
      break;
    }
    int64_t fr_end = esp_timer_get_time();

    int64_t frame_time = fr_end - last_frame;
    frame_time /= 1000;
    log_i(
      "MJPG: %uB %ums (%.1ffps), AVG: %ums (%.1ffps)", (uint32_t)(_jpg_buf_len), (uint32_t)frame_time, 1000.0 / (uint32_t)frame_time, avg_frame_time,
      1000.0 / avg_frame_time
    );

    // Create a JSON object for the line record data
    StaticJsonDocument<1024> record_json;
    JsonArray dataArray = record_json.createNestedArray("data");
    // consume up to 20 elements from the anchor line record queue
    line_record_t element;
    int count = 0;
    while (xQueueReceive(lenQueue, &element, 0) && count < 20) {
      count++;
      JsonObject obj = dataArray.createNestedObject();
      obj["time"] = element.time;
      obj["len"] = element.len;
    }
    serializeJson(record_json, json_buf);
    // boundary
    res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    // header for json
    if (res == ESP_OK) {
      size_t hlen = snprintf((char *)part_buf, 128, _STREAM_JSON_PART, strlen(json_buf));
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    // json
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)json_buf, strlen(json_buf));
    }
    if (res != ESP_OK) {
      log_e("Send json part failed");
      break;
    }
  }

  return res;
}

void startCameraServer(QueueHandle_t q) {
  lenQueue = q;

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL,
    .is_websocket = true,
    .handle_ws_control_frames = false,
    .supported_subprotocol = NULL
  };

  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.max_uri_handlers = 1;
  // config.server_port = 8888;
  // config.ctrl_port += 1;
  log_i("Starting stream server on port: '%d'", config.server_port);
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}