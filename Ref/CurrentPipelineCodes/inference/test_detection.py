#!/usr/bin/env python3
"""
Test script for debugging detection issues
Run this to test inference service without full GUI
"""
import zmq
import time
import numpy as np
import struct

def test_detection_service():
    """Send test commands to inference service."""

    print("🧪 Testing Inference Service")
    print("=" * 60)

    context = zmq.Context()

    # Connect to control socket
    print("Connecting to control socket (port 5560)...")
    control_socket = context.socket(zmq.REQ)
    control_socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10 second timeout
    control_socket.connect("tcp://localhost:5560")
    print("✅ Connected\n")

    # Test 1: Check service is responding
    print("Test 1: Ping service")
    print("-" * 60)
    try:
        control_socket.send_json({"cmd": "ping"})
        reply = control_socket.recv_json()
        print(f"Reply: {reply}\n")
    except zmq.error.Again:
        print("❌ Service not responding - is inference_service.py running?")
        return
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return

    # Test 2: Enable enhancement
    print("Test 2: Enable enhancement")
    print("-" * 60)
    try:
        control_socket.send_json({"cmd": "set_enhancement", "enabled": True})
        reply = control_socket.recv_json()
        print(f"Reply: {reply}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")

    # Test 3: Enable detection
    print("Test 3: Enable detection")
    print("-" * 60)
    print("Sending detection enable command...")
    print("Watch the inference service terminal for detailed output!\n")

    try:
        control_socket.send_json({
            "cmd": "set_detection",
            "enabled": True,
            "conf": 0.5,
            "model": "models/yolo_particles.pt"
        })

        print("Waiting for reply (may take 30+ seconds if loading model)...")
        reply = control_socket.recv_json()

        print("\nReply received:")
        print(f"  Status: {reply.get('status')}")
        print(f"  Detection enabled: {reply.get('detection')}")
        print(f"  YOLO available: {reply.get('yolo_available')}")
        print(f"  Model loaded: {reply.get('model_loaded')}")
        print(f"  Needs warmup: {reply.get('needs_warmup')}")
        print(f"  Model path: {reply.get('model')}")

        if reply.get('detection'):
            print("\n✅ Detection enabled successfully!")
            print("\nNext steps:")
            print("  1. Start camera service (if not running)")
            print("  2. Watch inference service terminal for warmup messages")
            print("  3. After warmup, frames should show green detection boxes")
        else:
            print("\n❌ Detection failed to enable")
            if not reply.get('yolo_available'):
                print("  Issue: ultralytics not installed")
                print("  Fix: conda activate holovision_infer && pip install ultralytics")
            elif not reply.get('model_loaded'):
                print("  Issue: Model file not found")
                print(f"  Expected: {reply.get('model')}")
                print("  Fix: Check model path and file existence")

    except zmq.error.Again:
        print("❌ Timeout waiting for reply (took > 10 seconds)")
        print("   This might mean:")
        print("   - Model is loading (check inference service terminal)")
        print("   - Service crashed (check for error messages)")
        print("   - Increase timeout if model is very large")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Disable detection
    print("\n" + "=" * 60)
    print("Test 4: Disable detection")
    print("-" * 60)
    try:
        control_socket.send_json({"cmd": "set_detection", "enabled": False})
        reply = control_socket.recv_json()
        print(f"Reply: {reply}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")

    print("=" * 60)
    print("Testing complete!")
    print("\nIf you see errors, check:")
    print("  1. Inference service is running: python inference_service.py")
    print("  2. Model file exists at: models/yolo_particles.pt")
    print("  3. ultralytics is installed: pip list | grep ultralytics")
    print("  4. GPU is available: python -c 'import torch; print(torch.cuda.is_available())'")


if __name__ == "__main__":
    test_detection_service()
