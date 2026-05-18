import os
from mtcnn_face_detection import FaceExtractorMTCNN


def main():
    # Set these via environment variables or edit directly:
    #   export DATA_DIR=/path/to/raw/dataset
    #   export PROCESSED_DIR=/path/to/output
    raw_data_dir       = os.environ.get("DATA_DIR",       os.path.join("data", "raw"))
    processed_data_dir = os.environ.get("PROCESSED_DIR",  os.path.join("data", "mtcnn_output"))

    extractor = FaceExtractorMTCNN(target_size=(256, 256))

    splits = ["train", "validation", "test"]

    for split in splits:
        print(f"\n{'=' * 60}")
        print(f"Processing {split}")
        print(f"{'=' * 60}")

        extractor.preprocess_dataset(
            input_dir=raw_data_dir,
            output_dir=processed_data_dir,
            split=split,
        )

    print("\nDONE")
    print(f"Saved at: {processed_data_dir}")


if __name__ == "__main__":
    main()