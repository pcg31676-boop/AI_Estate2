"""
Generate QR codes for the AI Estate Valuation website and NeoReader.
"""

import os
import socket
import qrcode
from PIL import Image

ARTIFACT_DIR = r"C:\Users\hp\.gemini\antigravity\brain\e98caaed-3706-4f9b-b973-351f03f69dfb"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "10.122.191.166"

local_ip = get_local_ip()
PORT = 8501

NETWORK_URL = f"http://{local_ip}:{PORT}"
LOCALHOST_URL = f"http://localhost:{PORT}"
NEOREADER_SCHEME_URL = f"neoreader://{local_ip}:{PORT}"

targets = [
    {
        "filename": "qrcode_network.png",
        "data": NETWORK_URL,
        "title": f"Mobile Network Wi-Fi URL ({NETWORK_URL})",
        "description": "Scan with NeoReader or any phone camera (device must be on the same Wi-Fi)"
    },
    {
        "filename": "qrcode_neoreader.png",
        "data": NEOREADER_SCHEME_URL,
        "title": f"NeoReader App Deep Link ({NEOREADER_SCHEME_URL})",
        "description": "Direct URI scheme for NeoReader application"
    },
    {
        "filename": "qrcode_localhost.png",
        "data": LOCALHOST_URL,
        "title": f"Localhost URL ({LOCALHOST_URL})",
        "description": "For scanning or testing directly on the host device"
    },
]

os.makedirs(ARTIFACT_DIR, exist_ok=True)

print(f"Generating QR Codes for Local IP: {local_ip}...")

for item in targets:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(item["data"])
    qr.make(fit=True)
    img = qr.make_image(fill_color="#0F172A", back_color="#FFFFFF")

    # Save to project directory
    proj_path = os.path.join(PROJECT_DIR, item["filename"])
    img.save(proj_path)

    # Save to artifact directory
    artifact_path = os.path.join(ARTIFACT_DIR, item["filename"])
    img.save(artifact_path)

    print(f"✔ Generated {item['filename']} -> {item['data']}")
