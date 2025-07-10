import cv2
import os



def initial_report(report: dict) -> None:
    
    report_name = report.get("features_name", "UNKNOWN ANALYSIS")
    print(f"\n[INITIAL REPORT] FROM {report_name.upper()}")
    print("-" * 30)

    for key, value in report.items():
        if key == "features_name":
            continue
        if not isinstance(value, str):
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    print("-" * 30)


    temp_dir = os.path.join("algorithm", "image_forensics", "temp_result")
    for file_name in os.listdir(temp_dir):
        if "output" in file_name.lower():
            img_path = os.path.join(temp_dir, file_name)
            img = cv2.imread(img_path)

            if img is not None:
                cv2.imshow(f"{file_name}", img)
                key = cv2.waitKey(0)
                if key == 27:
                    cv2.destroyAllWindows()