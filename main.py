import os
import argparse
import json
import logging
import time

from src.processing_pipeline import DocumentProcessor
from src.postprocessing import postprocess_output

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Document Processing Pipeline")
    parser.add_argument("--input_dir", type=str, default="input", help="Directory containing documents to process")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save JSON output files")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO",
                        help="Set logging verbosity (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if not os.path.exists(args.input_dir):
        logger.error("Input directory not found: %s", args.input_dir)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Initializing DocumentProcessor...")
    start_time = time.time()
    try:
        processor = DocumentProcessor()
    except Exception as e:
        logger.error("Failed to initialize models. Ensure you have a GPU and dependencies installed.\nError: %s", e)
        return

    VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.webp', '.pdf'}
    all_files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    files = [f for f in all_files if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
    skipped = len(all_files) - len(files)
    if skipped:
        logger.info("Skipped %d non-document files (e.g. .gitkeep)", skipped)
    logger.info("Found %d document files in %s", len(files), args.input_dir)

    for filename in files:
        file_path = os.path.join(args.input_dir, filename)
        logger.info("Processing %s...", filename)

        output_filename = os.path.splitext(filename)[0] + ".json"
        output_file_path = os.path.join(args.output_dir, output_filename)

        try:
            doc_data, page_images = processor.process_document(file_path)
            doc_data = postprocess_output(doc_data, page_images=page_images,
                                          handwriting_recognizer=processor.handwriting_recognizer)
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Error processing %s: %s", filename, e, exc_info=True)
            error_data = {"filename": filename, "error": str(e)}
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)

    logger.info("Processing complete in %.2fs. Results saved to %s", time.time() - start_time, args.output_dir)


if __name__ == "__main__":
    main()
