import cv2
import pyttsx3
import time

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    # Try default camera index 0. If you have an external webcam, try 1,2,...
    cam_index = 0
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # use CAP_DSHOW on Windows to avoid warnings

    if not cap.isOpened():
        print(f"ERROR: Cannot open webcam at index {cam_index}.")
        print("If you have another camera, change cam_index to 1 or 2 in the code and try again.")
        return

    # Speak a test phrase (offline)
    print("Speaking test phrase...")
    speak("Hello. This is your blind navigation assistant. Webcam and audio are working.")
    time.sleep(0.5)

    print("Press 's' to speak sample alert, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read from camera. Exiting.")
            break

        # Resize for display (optional)
        display = cv2.resize(frame, (640, 480))

        # Show frame
        cv2.imshow("Webcam - Blind Navigation Assistant (press q to quit)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            speak("Warning. Obstacle detected ahead. Please be careful.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
