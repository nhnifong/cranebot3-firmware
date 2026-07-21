"""Opens the default camera and continuously prints the data of any QR code it sees.

Shows a preview window so you can aim the camera; press 'q' or Esc to quit.
"""
import sys
import cv2
from pyzbar.pyzbar import decode

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera", file=sys.stderr)
        sys.exit(1)

    last_seen = set()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray_view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            decoded_objects = decode(gray_view)

            seen = set()
            for obj in decoded_objects:
                qr_data = obj.data.decode('utf-8')
                seen.add(qr_data)
                if qr_data not in last_seen:
                    print(qr_data)
                cv2.rectangle(frame, (obj.rect.left, obj.rect.top),
                              (obj.rect.left + obj.rect.width, obj.rect.top + obj.rect.height),
                              (0, 255, 0), 2)
            last_seen = seen

            cv2.imshow('QR code reader', frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
