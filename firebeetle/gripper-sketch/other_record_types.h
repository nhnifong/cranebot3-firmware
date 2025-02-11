typedef struct {
  float time; // unix seconds
  float len; // meters
} line_record_t;

// acceleration in global reference frame, without gravity 
typedef struct {
  float time; // unix seconds
  float x;
  float y; // up
  float z;
} accel_record_t;
