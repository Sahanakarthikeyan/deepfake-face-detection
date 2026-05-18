import os
import cv2
import numpy as np
from tqdm import tqdm
from mtcnn import MTCNN


class FaceExtractorMTCNN:

    def __init__(self, target_size=(256, 256)):
        self.detector    = MTCNN()
        self.target_size = target_size

    def extract_face(self, image_path):
        """
        Detects and extracts the largest face from an image.

        Args:
            image_path (str): Path to input image.

        Returns:
            numpy.ndarray: Normalized extracted face image, or None if no face found.
        """
        try:
            img = cv2.imread(image_path)

            if img is None:
                return None

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_img, w_img, _ = img.shape

            results = self.detector.detect_faces(img_rgb)

            if len(results) == 0:
                return None

            # Take largest detected face
            face = max(results, key=lambda x: x['box'][2] * x['box'][3])
            x, y, w, h = face['box']

            # Fix negative values
            x = max(0, x)
            y = max(0, y)

            # Fix overflow beyond image size
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)

            face_crop = img_rgb[y:y2, x:x2]

            if face_crop.size == 0:
                return None

            # Resize
            face_resized = cv2.resize(face_crop, self.target_size)

            # Normalize
            face_normalized = face_resized.astype(np.float32) / 255.0

            return face_normalized

        except Exception as e:
            print(f"Error: {image_path} -> {e}")
            return None

    def preprocess_dataset(self, input_dir, output_dir, split):

        classes = ['real', 'fake']

        stats = {
            cls: {'total': 0, 'extracted': 0, 'failed': 0}
            for cls in classes
        }

        for cls in classes:

            input_class_dir  = os.path.join(input_dir, split, cls)
            output_class_dir = os.path.join(output_dir, split, cls)

            os.makedirs(output_class_dir, exist_ok=True)

            if not os.path.exists(input_class_dir):
                print(f"Missing: {input_class_dir}")
                continue

            images = [
                f for f in os.listdir(input_class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            stats[cls]['total'] = len(images)

            print(f"\nProcessing {split}/{cls}: {len(images)} images")

            for img_file in tqdm(images):

                in_path  = os.path.join(input_class_dir, img_file)
                out_path = os.path.join(output_class_dir, img_file)

                face = self.extract_face(in_path)

                if face is not None:
                    face_uint8 = (face * 255).astype(np.uint8)
                    cv2.imwrite(out_path, cv2.cvtColor(face_uint8, cv2.COLOR_RGB2BGR))
                    stats[cls]['extracted'] += 1
                else:
                    stats[cls]['failed'] += 1

        # Summary
        print("\n" + "=" * 50)
        print(f"{split.upper()} SUMMARY")
        print("=" * 50)

        for cls in classes:
            total     = stats[cls]['total']
            extracted = stats[cls]['extracted']
            failed    = stats[cls]['failed']
            print(f"{cls}: {extracted}/{total} extracted | Failed: {failed}")

        return stats