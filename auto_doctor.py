import requests
import time
import os

# ==========================================
# CONFIGURATION
# ==========================================
ESP32_IP = "http://192.168.0.116"
ESP32_CAPTURE_URL = f"{ESP32_IP}/capture"
ESP32_CONTROL_URL = f"{ESP32_IP}/control"

SERVER_API_URL = "http://127.0.0.1:5000/api/iot/upload"
API_KEY = "greeneye_secret_pass_123"

BACKUP_FOLDER = r"E:\IOT\leaf_backups"
os.makedirs(BACKUP_FOLDER, exist_ok=True)

# ==========================================
# SETUP
# ==========================================
print("--- IoT Plant Doctor: Batch Mode ---")

# 1. Arm the Flash
try:
    print("Configuring Flash to MAX...")
    requests.get(f"{ESP32_CONTROL_URL}?var=led_intensity&val=255", timeout=5)
    print("✅ Flash Ready.")
except:
    print("⚠️ Warning: Could not configure flash (ESP32 might be offline).")

# 2. Set User
print("Who should receive these images?")
target_email = input("Enter your registered Email Address: ").strip()
if not target_email:
    print("Error: Email is required.")
    exit()

print(f"✅ Target Locked: {target_email}")
time.sleep(1)

# ==========================================
# MAIN BATCH LOOP
# ==========================================
while True:
    print("\n" + "="*40)
    print("   READY TO CAPTURE")
    print("="*40)
    user_input = input("👉 Press [ENTER] to take 5 photos (or type 'q' to quit): ")

    if user_input.lower() == 'q':
        print("Exiting...")
        break

    print("\n🚀 Starting Batch of 5 Photos...")

    # --- THE 5-PHOTO LOOP ---
    for i in range(1, 6):
        print(f"\n📸 [Photo {i}/5] Capturing...")

        try:
            # 1. Capture
            esp_response = requests.get(ESP32_CAPTURE_URL, timeout=10)
            
            if esp_response.status_code == 200:
                # 2. Save
                timestamp = int(time.time())
                filename = f"leaf_batch_{timestamp}_{i}.jpg"
                filepath = os.path.join(BACKUP_FOLDER, filename)

                with open(filepath, "wb") as f:
                    f.write(esp_response.content)

                # 3. Upload
                print(f"    Uploading to Dashboard...")
                with open(filepath, "rb") as img_file:
                    files = {'image': img_file}
                    data = {'email': target_email} 
                    headers = {'X-API-KEY': API_KEY}
                    
                    try:
                        api_response = requests.post(SERVER_API_URL, files=files, data=data, headers=headers)
                        if api_response.status_code == 200:
                            res = api_response.json()
                            print(f"    ✅ SUCCESS! Diagnosis: {res.get('health')} - {res.get('disease')}")
                        else:
                            print(f"    ❌ Server Error: {api_response.text}")
                    except Exception as e:
                        print(f"    ❌ Upload Failed: {e}")
            else:
                print(f"    ❌ ESP32 Error: {esp_response.status_code}")

        except Exception as e:
            print(f"    ❌ Connection Error: {e}")

        # Wait 5 seconds between shots to let flash recharge
        if i < 5: 
            time.sleep(5)

    print("\n✅ Batch Complete!")