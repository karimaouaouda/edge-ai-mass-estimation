"""Regression tests for Jetson native-library import order."""

import subprocess
import sys


def test_importing_cli_does_not_eagerly_load_vision_or_ml_native_stacks():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; import edge_ai_mass.cli; "
            "print('cv2' in sys.modules, 'numpy' in sys.modules, "
            "'sklearn' in sys.modules, 'torch' in sys.modules)"
        ),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False False False"
