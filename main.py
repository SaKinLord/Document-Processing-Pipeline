
import os
import argparse
import json
import time
from src.processing_pipeline import DocumentProcessor

def main():
    parser = argparse.ArgumentParser(description="Document Processing Pipeline")
    parser.add_argument("--input_dir", type=str, default="input", help="Directory containing documents to process")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save JSON output files")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Input directory not found: {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("Initializing DocumentProcessor...")
    start_time = time.time()
    try:
        processor = DocumentProcessor()
    except Exception as e:
        print(f"Failed to initialize models. Ensure you have a GPU and dependencies installed.\nError: {e}")
        return

    files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    # EMERGENCY FIX: TARGET ONLY 87428306.png
    files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    print(f"Found {len(files)} files in {args.input_dir}")

    for filename in files:
        file_path = os.path.join(args.input_dir, filename)
        print(f"Processing {filename}...")
        
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_file_path = os.path.join(args.output_dir, output_filename)

        try:
            doc_data = processor.process_document(file_path)
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing {filename}: {e}")
            # Write error to a specific error file for this logical unit, or skipping. 
            # For robustness, we can write an error JSON or just log it. 
            # Given the requirement for robustness, writing an error JSON is helpful.
            error_data = {"filename": filename, "error": str(e)}
            with open(output_file_path, "w", encoding="utf-8") as f:
                 json.dump(error_data, f, indent=2, ensure_ascii=False)

    print(f"Processing complete in {time.time() - start_time:.2f}s. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
