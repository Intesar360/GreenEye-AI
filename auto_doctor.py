import requests
import time
import os

# ==========================================
# CONFIGURATION
# ==========================================
ESP32_CAM_URL = "http://192.168.0.120/capture"
SERVER_API_URL = "http://127.0.0.1:5000/api/iot/upload"
API_KEY = "greeneye_secret_pass_123"
BACKUP_FOLDER = r"E:\IOT\leaf_backups"
os.makedirs(BACKUP_FOLDER, exist_ok=True)

# ==========================================
# USER LOGIN (SIMULATED)
# ==========================================
print("--- IoT Automation Started ---")
print("Who should receive these images?")
target_email = input("Enter your registered Email Address: ").strip()

if not target_email:
    print("Error: Email is required to start.")
    exit()

print(f"✅ OK! Sending images to: {target_email}")
print("Starting capture loop...")
time.sleep(2)

# ==========================================
# AUTOMATION LOOP
# ==========================================
while True:
    try:
        print(f"\n[1] Requesting image for {target_email}...")
        
        try:
            esp_response = requests.get(ESP32_CAM_URL, timeout=5)
        except:
            print("    Connection Error: Cannot reach ESP32.")
            time.sleep(5)
            continue

        if esp_response.status_code == 200:
            # Save locally
            timestamp = int(time.time())
            filename = f"leaf_{timestamp}.jpg"
            filepath = os.path.join(BACKUP_FOLDER, filename)

            with open(filepath, "wb") as f:
                f.write(esp_response.content)

            # Upload to Flask Server
            print("    [2] Uploading to Dashboard...")
            
            with open(filepath, "rb") as img_file:
                files = {'image': img_file}
                # HERE IS THE CHANGE: We send the email along with the image
                data = {'email': target_email} 
                headers = {'X-API-KEY': API_KEY}
                
                try:
                    api_response = requests.post(SERVER_API_URL, files=files, data=data, headers=headers)
                    
                    if api_response.status_code == 200:
                        res = api_response.json()
                        print(f"    SUCCESS! Uploaded to {res.get('user')}")
                        print(f"    Diagnosis: {res.get('health')} - {res.get('disease')}")
                    elif api_response.status_code == 404:
                        print("    ERROR: That email does not exist in the database!")
                        print("    Please stop (Ctrl+C) and restart with a valid email.")
                    else:
                        print(f"    Server Error: {api_response.text}")
                        
                except Exception as e:
                    print(f"    Upload Failed: {e}")

        else:
            print(f"    ESP32 returned error: {esp_response.status_code}")

    except Exception as e:
        print(f"    Critical Error: {e}")

    print("    Waiting 10 seconds...")
    time.sleep(10)