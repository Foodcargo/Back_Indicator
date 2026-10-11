import cv2


def main():
    print("Scanning camera indexes 0 through 9...")

    found = []
    for index in range(10):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        ok, frame = cap.read() if cap.isOpened() else (False, None)

        if ok and frame is not None:
            height, width = frame.shape[:2]
            found.append(index)
            print(f"[OK] index {index}: {width}x{height}")
        else:
            print(f"[--] index {index}: not available")

        cap.release()

    if found:
        print()
        print("Usable camera indexes:", ", ".join(map(str, found)))
        print("Try the USB camera with: python pose_analysis.py --camera <index>")
    else:
        print()
        print("No usable camera was found.")


if __name__ == "__main__":
    main()
