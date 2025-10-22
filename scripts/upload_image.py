#!/usr/bin/env python3
"""
Image Upload Script

Uploads image files to OSS using upload_g2.py

Usage:
    python upload_image.py <image_path> [--type image|video|auto]

Arguments:
    image_path: Path to the local image file to upload

Options:
    --type: File type (image, video, auto). Default: auto

Examples:
    python upload_image.py /path/to/image.png
    python upload_image.py /path/to/video.mp4 --type video
    python upload_image.py /path/to/file.jpg --type auto
"""

import os
import sys
import argparse
import importlib.util

def load_upload_g2():
    """Load the upload_g2 module"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    upload_g2_path = os.path.join(script_dir, "upload_g2.py")
    
    spec = importlib.util.spec_from_file_location("upload_g2", upload_g2_path)
    upload_g2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upload_g2)
    return upload_g2

def upload_image(image_path, file_type='auto'):
    """Upload an image file using upload_g2.py"""
    upload_g2 = load_upload_g2()

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        return None

    # Check file size
    file_size = os.path.getsize(image_path)
    print(f"📁 File: {os.path.basename(image_path)}")
    print(f"📊 Size: {file_size / 1024 / 1024:.2f} MB")
    print(f"🏷️  Type: {file_type}")
    print()

    # Upload the file
    print("🚀 Starting upload...")
    oss_url = upload_g2.upload_url(image_path, file_type=file_type)

    if oss_url:
        print(f"\n✅ Upload successful!")
        print(f"🔗 OSS URL: {oss_url}")
        return oss_url
    else:
        print("\n❌ Upload failed!")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Upload image files to OSS using upload_g2.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_image.py image.png
  python upload_image.py video.mp4 --type video
  python upload_image.py /path/to/photo.jpg --type auto
        """
    )

    parser.add_argument(
        "image_path",
        help="Path to the image/video file to upload"
    )

    parser.add_argument(
        "--type",
        choices=['image', 'video', 'auto'],
        default='auto',
        help="File type (default: auto)"
    )

    args = parser.parse_args()

    print("🖼️  Image Upload Script")
    print("=" * 50)

    try:
        result = upload_image(args.image_path, args.type)
        if result:
            print("\n" + "=" * 50)
            print("🎉 Upload completed successfully!")
            print(f"📋 Final OSS URL: {result}")
            sys.exit(0)
        else:
            print("\n" + "=" * 50)
            print("💥 Upload failed!")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


