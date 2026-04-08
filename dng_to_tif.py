import os
import glob
import argparse
import subprocess

try:
    import rawpy
    import imageio
except ImportError:
    print("ERROR: Missing required Python libraries.")
    print("Please run: pip install rawpy imageio")
    exit(1)

def batch_convert_dng_to_tif(input_dir, output_dir, copy_exif=True):
    """
    Converts all .dng images in the input directory to .tif in the output directory.
    Optionally relies on ExifTool to copy XMP/EXIF telemetry for the Andvari pipeline.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dng_files = glob.glob(os.path.join(input_dir, "*.[dD][nN][gG]"))
    
    if not dng_files:
        print(f"No .dng files found in {input_dir}")
        return

    print(f"Found {len(dng_files)} .dng files. Starting batch conversion...")

    for i, dng_path in enumerate(dng_files, start=1):
        basename = os.path.basename(dng_path)
        name, _ = os.path.splitext(basename)
        tif_path = os.path.join(output_dir, f"{name}.tif")
        
        print(f"[{i}/{len(dng_files)}] Processing: {basename} -> {name}.tif")

        try:
            # 1. Convert the RAW DNG pixels to RGB
            with rawpy.imread(dng_path) as raw:
                # use_camera_wb applies the drone's native white balance
                # output_bps=16 keeps it as a 16-bit uncompressed image
                rgb_image = raw.postprocess(use_camera_wb=True, output_bps=16)

            # 2. Save out as an uncompressed TIFF
            imageio.imsave(tif_path, rgb_image, format='TIFF')

            # 3. Copy Metadata (CRITICAL FOR ANDVARI PIPELINE)
            if copy_exif:
                try:
                    # -TagsFromFile copies all tags, -overwrite_original prevents creating backup files
                    subprocess.run(
                        ["exiftool", "-TagsFromFile", dng_path, "-all:all>all:all", "-overwrite_original", tif_path],
                        check=True,
                        capture_output=True
                    )
                except FileNotFoundError:
                    print(f"  [WARNING] 'exiftool' not found. Telemetry was stripped from {name}.tif!")
                    print(f"  [WARNING] Please install exiftool and add it to PATH.")
                except subprocess.CalledProcessError as e:
                    print(f"  [WARNING] ExifTool failed to transfer metadata: {e}")

        except Exception as e:
            print(f"  [ERROR] Failed converting {basename}: {e}")

    print("\n--- Batch Conversion Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert DNG to TIF and retain telemetry.")
    parser.add_argument("--input", required=True, help="Directory containing raw .dng images")
    parser.add_argument("--output", required=True, help="Directory to save the finished .tif images")
    parser.add_argument("--no-exif", action="store_true", help="Skip copying EXIF data (WARNING: Breaks Cartographer)")
    
    args = parser.parse_args()
    
    batch_convert_dng_to_tif(args.input, args.output, copy_exif=not args.no_exif)
