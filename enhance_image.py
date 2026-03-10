import subprocess
import os

def enhance_image(input_path, output_path):

    os.makedirs("enhanced_output", exist_ok=True)

    command = [
        "python",
        "inference_realesrgan.py",
        "-i", input_path,
        "-o", "../enhanced_output",
        "-n", "realesr-general-x4v3",
        "--fp32"
    ]

    subprocess.run(
        command,
        cwd="Real-ESRGAN",
        check=True
    )

    filename = os.path.basename(input_path)
    enhanced_file = next(
        os.path.join("enhanced_output", f)
        for f in os.listdir("enhanced_output")
        if f.startswith(os.path.basename(input_path).split(".")[0])
    )

    os.rename(enhanced_file, output_path)

    return output_path