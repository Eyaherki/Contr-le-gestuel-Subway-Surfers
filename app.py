import cv2
import mediapipe as mp
import time
import keyboard
import pygetwindow as gw

# Nom de la fenêtre BlueStacks (à adapter selon ta config)
BLUESTACKS_WINDOW_TITLE = "BlueStacks"

# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture webcam
cap = cv2.VideoCapture(0)

prev_action = ""
last_time = time.time()

def detect_gesture(lm_list):
    x = lm_list[8][0]  # X de l'index
    y = lm_list[8][1]  # Y de l'index

    if x < 100:
        return "left"
    elif x > 250:
        return "right"
    elif y < 150:
        return "jump"
    return ""

def activate_bluestacks():
    windows = gw.getWindowsWithTitle(BLUESTACKS_WINDOW_TITLE)
    if windows:
        window = windows[0]
        if not window.isActive:
            window.activate()
            time.sleep(0.1)  # Attendre que la fenêtre soit active
        return True
    else:
        print("Fenêtre BlueStacks non trouvée.")
        return False

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            h, w, _ = img.shape
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = detect_gesture(lm_list)

            if gesture != prev_action and time.time() - last_time > 1:
                if activate_bluestacks():
                    if gesture == "left":
                        keyboard.press_and_release("left")
                        print("← LEFT")
                    elif gesture == "right":
                        keyboard.press_and_release("right")
                        print("→ RIGHT")
                    elif gesture == "jump":
                        keyboard.press_and_release("up")
                        print("↑ JUMP")

                    prev_action = gesture
                    last_time = time.time()

    cv2.imshow("Hand Gesture Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
