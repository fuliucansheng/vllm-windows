# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm.utils import windows_utils


class TestWindowsCudaRuntime(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cuda path ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def dll(self, relative, machine=43620):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        header = bytearray(128)
        header[:2] = b"MZ"
        struct.pack_into("<I", header, 60, 64)
        header[64:68] = b"PE\x00\x00"
        struct.pack_into("<H", header, 68, machine)
        path.write_bytes(header)
        return str(path.resolve())

    def test_native_arm64_avoids_x64(self):
        expected = self.dll("bin/arm64/cudart64_13.dll")
        self.dll("bin/x64/cudart64_13.dll", 34404)
        with (
            patch.dict(os.environ, CUDA_PATH=str(self.root)),
            patch.object(
                windows_utils.sysconfig, "get_platform", return_value="win-arm64"
            ),
        ):
            self.assertEqual(windows_utils.find_cuda_runtime("13.4"), expected)

    def test_x64_process_on_arm64_host(self):
        expected = self.dll("bin/x64/cudart64_13.dll", 34404)
        self.dll("bin/arm64/cudart64_13.dll")
        with (
            patch.dict(
                os.environ, CUDA_HOME=str(self.root), PROCESSOR_ARCHITECTURE="ARM64"
            ),
            patch.object(
                windows_utils.sysconfig, "get_platform", return_value="win-amd64"
            ),
        ):
            self.assertEqual(windows_utils.find_cuda_runtime("13.4"), expected)

    def test_legacy_flat_layout(self):
        expected = self.dll("bin/cudart64_110.dll", 34404)
        with (
            patch.dict(os.environ, CUDA_ROOT=str(self.root)),
            patch.object(
                windows_utils.sysconfig, "get_platform", return_value="win-amd64"
            ),
        ):
            self.assertEqual(windows_utils.find_cuda_runtime("11.8"), expected)

    def test_nvcc_uses_parent_directory(self):
        expected = self.dll("bin/arm64/cudart64_13.dll")
        compiler = self.dll("bin/arm64/nvcc.exe")
        with (
            patch.object(windows_utils, "which", return_value=compiler),
            patch.object(
                windows_utils.sysconfig, "get_platform", return_value="win-arm64"
            ),
        ):
            self.assertEqual(windows_utils.find_cuda_runtime("13.4"), expected)

    def test_explicit_override(self):
        expected = self.dll("custom/runtime.dll")
        with patch.dict(os.environ, VLLM_CUDART_SO_PATH=expected):
            self.assertEqual(windows_utils.find_cuda_runtime("13.4"), expected)

    def test_missing_runtime_does_not_return_invented_path(self):
        self.dll("bin/x64/cudart64_13.dll", 34404)
        with (
            patch.dict(os.environ, CUDA_PATH=str(self.root)),
            patch.object(
                windows_utils.sysconfig, "get_platform", return_value="win-arm64"
            ),
            self.assertRaisesRegex(FileNotFoundError, "win-arm64"),
        ):
            windows_utils.find_cuda_runtime("13.4")

    def test_missing_override_fails(self):
        with (
            patch.dict(os.environ, VLLM_CUDART_SO_PATH=str(self.root / "missing")),
            self.assertRaisesRegex(FileNotFoundError, "VLLM_CUDART_SO_PATH"),
        ):
            windows_utils.find_cuda_runtime("13.4")

    def test_other_architecture_nvcc_uses_correct_sibling(self):
        expected = self.dll("bin/arm64/cudart64_13.dll")
        compiler = self.dll("bin/x64/nvcc.exe", 34404)
        self.dll("bin/x64/cudart64_13.dll", 34404)
        with (
            patch.object(windows_utils, "which", return_value=compiler),
            patch.object(
                windows_utils.sysconfig, "get_platform", return_value="win-arm64"
            ),
        ):
            self.assertEqual(windows_utils.find_cuda_runtime("13.4"), expected)

    def test_unlabelled_wrong_architecture_runtime_fails(self):
        compiler = self.dll("tools/nvcc.exe", 34404)
        self.dll("tools/cudart64_13.dll", 34404)
        with (
            patch.object(windows_utils, "which", return_value=compiler),
            patch.object(
                windows_utils.sysconfig, "get_platform", return_value="win-arm64"
            ),
            self.assertRaisesRegex(FileNotFoundError, "PE machine 0x8664"),
        ):
            windows_utils.find_cuda_runtime("13.4")

    def test_explicit_override_is_not_replaced_by_discovery(self):
        expected = self.dll("explicit.dll", 34404)
        with (
            patch.dict(os.environ, VLLM_CUDART_SO_PATH=expected),
            patch.object(
                windows_utils.sysconfig, "get_platform", return_value="win-arm64"
            ),
        ):
            self.assertEqual(windows_utils.find_cuda_runtime("13.4"), expected)

    @unittest.skipUnless(os.name == "nt", "Windows loader integration")
    def test_find_loaded_library_returns_discovered_cudart(self):
        from vllm.utils.system_utils import find_loaded_library

        expected = self.dll("explicit/runtime.dll")
        with patch.dict(os.environ, VLLM_CUDART_SO_PATH=expected):
            actual = find_loaded_library("libcudart")
            self.assertEqual(actual, expected)
            self.assertEqual(os.environ["VLLM_CUDART_SO_PATH"], expected)
