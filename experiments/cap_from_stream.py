import os
import subprocess

def capture_stream_continuous(stream_url, output_dir, total_images=100):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Connecting to {stream_url}...")
    print(f"Extracting {total_images} images at 1 frame per second...")

    # Pattern for file names: img_001.jpg, img_002.jpg, etc.
    output_pattern = os.path.join(output_dir, "img_%03d.jpg")
    
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'mpegts',
        '-i', stream_url,
        '-vf', 'fps=1',           # Video Filter: Extract exactly 1 frame per second
        '-vframes', str(total_images), # Stop completely after exporting 100 frames
        '-f', 'image2',
        output_pattern
    ]
    
    try:
        # Run natively and let FFmpeg print its progress naturally
        # You'll see frame counts incrementing in real-time
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if process.returncode == 0:
            print("\nSuccess! All 100 images captured cleanly.")
        else:
            print("\n--- [FFMPEG ERROR] ---")
            print(process.stderr)
            print("----------------------")
            
    except KeyboardInterrupt:
        print("\nCapture interrupted by user.")

if __name__ == "__main__":
    # Note: I kept port 8889 from your last error log, change back to 8888 if needed!
    STREAM_URL = "tcp://192.168.1.226:8888" 
    OUTPUT_DIRECTORY = "images/cap/"
    
    capture_stream_continuous(STREAM_URL, OUTPUT_DIRECTORY, total_images=100)