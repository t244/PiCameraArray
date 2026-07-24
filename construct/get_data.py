#!/usr/bin/env python3
"""
Data gathering script for PiCameraArray.
Collects data from all 16 Raspberry Pis (e00 to e15).

Transport is platform-dependent:
  - Windows: plink / pscp (PuTTY tools, same as Invoke-PiCommand)
  - Linux/macOS: sshpass + ssh / rsync
"""

import subprocess
import os
import sys
import platform
import shutil
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
PI_NAMES = [f"e{i:02d}" for i in range(16)]  # e00 to e15
REMOTE_DATA_DIR_SSD = "/media/pi/HIKSEMI/data"     # External SSD mount point
REMOTE_DATA_DIR_LOCAL = "/home/pi/PiCameraArray/data"  # SD card fallback

SSH_PASSWORD = "pi"
IS_WINDOWS = platform.system() == "Windows"
# plink resolves bare hostnames (e00); ssh on Linux needs mDNS (.local)
HOST_SUFFIX = "" if IS_WINDOWS else ".local"


def pi_host(pi_name):
    return f"pi@{pi_name}{HOST_SUFFIX}"


def ssh_cmd(pi_name, remote_command):
    """Build a remote-command invocation for the current platform."""
    if IS_WINDOWS:
        return ["plink", "-pw", SSH_PASSWORD, "-batch",
                pi_host(pi_name), remote_command]
    return ["sshpass", "-p", SSH_PASSWORD,
            "ssh", "-o", "StrictHostKeyChecking=no",
            pi_host(pi_name), remote_command]


def find_base_directory():
    """Local base directory for collected data."""
    workspace_root = Path.cwd()
    local_path = workspace_root / "collected_data"
    print(f"Local collection directory: {local_path}")
    return local_path


LOCAL_BASE_DIR = find_base_directory()


def run_command(cmd, check=True):
    """Execute a command list and return CompletedProcess."""
    if isinstance(cmd, str):
        cmd = cmd.split()

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if check and result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Error: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)

    return result


def get_remote_data_dir(pi_name):
    """Determine which data directory exists on the Pi (SSD or SD)."""
    try:
        cmd = ssh_cmd(
            pi_name,
            f"test -d {REMOTE_DATA_DIR_SSD} && echo ssd || echo local")
        result = run_command(cmd, check=True)
        if result.stdout.strip() == "ssd":
            return REMOTE_DATA_DIR_SSD
        return REMOTE_DATA_DIR_LOCAL
    except Exception as e:
        print(f"  Warning: Could not determine data dir on {pi_name}, "
              f"using SD card: {e}")
        return REMOTE_DATA_DIR_LOCAL


def get_latest_directory(pi_name, remote_data_dir):
    """Get the newest session directory name on a Pi."""
    try:
        cmd = ssh_cmd(pi_name, f"ls -t {remote_data_dir} | head -1")
        result = run_command(cmd, check=True)
        latest_dir = result.stdout.strip()
        if latest_dir:
            return latest_dir
        print(f"Warning: No data directory found on {pi_name} "
              f"at {remote_data_dir}")
        return None
    except Exception as e:
        print(f"Error getting latest directory from {pi_name}: {e}")
        return None


DATA_EXTS = (".png", ".npz")  # single frames / burst archives


def list_remote_data_files(pi_name, remote_dir):
    """Return a set of data filenames (png/npz) in a remote directory."""
    cmd = ssh_cmd(pi_name,
                  f"ls {remote_dir} | grep -E '\\.(png|npz)$' || true")
    result = run_command(cmd, check=True)
    return {name for name in result.stdout.split() if name}


def local_data_files(local_dir_path):
    return {p.name for p in local_dir_path.iterdir()
            if p.suffix in DATA_EXTS} if local_dir_path.exists() else set()


def copy_data_from_pi(pi_name, remote_dir_name, remote_data_dir, local_dir):
    """Copy a whole session directory from a Pi (first-time copy)."""
    try:
        remote = f"{remote_data_dir}/{remote_dir_name}"
        local_path = local_dir / f"{pi_name}_{remote_dir_name}"
        local_path.mkdir(parents=True, exist_ok=True)

        print(f"Copying from {pi_name}... ({remote_dir_name})")

        if IS_WINDOWS:
            cmd = ["pscp", "-pw", SSH_PASSWORD, "-q", "-r",
                   f"{pi_host(pi_name)}:{remote}/*", str(local_path)]
        else:
            cmd = ["sshpass", "-p", SSH_PASSWORD,
                   "rsync", "-a",
                   "-e", "ssh -o StrictHostKeyChecking=no",
                   f"{pi_host(pi_name)}:{remote}/", str(local_path) + "/"]

        run_command(cmd, check=True)
        print(f"  ✓ Successfully copied to {local_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error copying from {pi_name}: {e}")
        return False


def copy_missing_files_from_pi(pi_name, remote_dir_name, remote_data_dir,
                               local_dir_path):
    """Copy only PNG files that are not present locally."""
    try:
        remote = f"{remote_data_dir}/{remote_dir_name}"
        print(f"Syncing missing files from {pi_name}... ({remote_dir_name})")

        if IS_WINDOWS:
            remote_files = list_remote_data_files(pi_name, remote)
            local_files = local_data_files(local_dir_path)
            missing = sorted(remote_files - local_files)
            for name in missing:
                cmd = ["pscp", "-pw", SSH_PASSWORD, "-q",
                       f"{pi_host(pi_name)}:{remote}/{name}",
                       str(local_dir_path)]
                run_command(cmd, check=True)
            print(f"  ✓ Copied {len(missing)} missing file(s) "
                  f"to {local_dir_path}")
        else:
            cmd = ["sshpass", "-p", SSH_PASSWORD,
                   "rsync", "-a", "--ignore-existing",
                   "-e", "ssh -o StrictHostKeyChecking=no",
                   f"{pi_host(pi_name)}:{remote}/",
                   str(local_dir_path) + "/"]
            run_command(cmd, check=True)
            print(f"  ✓ Successfully synced missing files "
                  f"to {local_dir_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error syncing missing files from {pi_name}: {e}")
        return False


def get_already_copied_pis(local_dir):
    """Check which Pis already have local data directories."""
    already_copied = set()
    if not local_dir.exists():
        return already_copied

    for item in local_dir.iterdir():
        if item.is_dir() and "_" in item.name:
            pi_name = item.name.split("_")[0]
            if pi_name in PI_NAMES:
                already_copied.add(pi_name)

    return already_copied


def count_remote_png_files(pi_name, remote_dir_name, remote_data_dir):
    """Count PNG files in the remote session directory."""
    try:
        cmd = ssh_cmd(
            pi_name,
            f"find {remote_data_dir}/{remote_dir_name} "
            f"\\( -name '*.png' -o -name '*.npz' \\) | wc -l")
        result = run_command(cmd, check=True)
        return int(result.stdout.strip())
    except Exception as e:
        print(f"  Error counting remote data files on {pi_name}: {e}")
        return -1


def count_local_png_files(local_dir_path):
    """Count data files (png/npz) in a local directory recursively."""
    if not local_dir_path.exists():
        return 0
    return sum(1 for p in local_dir_path.rglob("*") if p.suffix in DATA_EXTS)


def is_copy_complete(pi_name, remote_dir_name, remote_data_dir,
                     local_dir_path):
    """Compare remote and local PNG counts."""
    remote_count = count_remote_png_files(
        pi_name, remote_dir_name, remote_data_dir)
    local_count = count_local_png_files(local_dir_path)
    if remote_count < 0:
        return False, remote_count, local_count
    return remote_count == local_count, remote_count, local_count


def check_tools():
    """Verify the required transfer tools are available."""
    required = ["plink", "pscp"] if IS_WINDOWS else ["sshpass", "rsync"]
    missing = [t for t in required if shutil.which(t) is None]
    if missing:
        print(f"Error: required tool(s) not found in PATH: "
              f"{', '.join(missing)}")
        if IS_WINDOWS:
            print("Install PuTTY (provides plink/pscp): "
                  "winget install PuTTY.PuTTY")
        else:
            print("Install with your package manager, e.g.: "
                  "sudo apt install sshpass rsync")
        sys.exit(1)


def main():
    """Main function to orchestrate data gathering."""
    print("=" * 70)
    print("PiCameraArray Data Gathering Script")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: "
          f"{'Windows (plink/pscp)' if IS_WINDOWS else 'POSIX (sshpass/rsync)'}")
    print(f"Base directory: {LOCAL_BASE_DIR}\n")

    check_tools()

    # Step 1: Get latest directory from e00
    print("Step 1: Getting latest directory from e00...")
    e00_remote_dir = get_remote_data_dir("e00")
    print(f"  Using data directory on e00: {e00_remote_dir}")

    e00_latest = get_latest_directory("e00", e00_remote_dir)
    if not e00_latest:
        print("Error: Could not get latest directory from e00. Aborting.")
        sys.exit(1)
    print(f"  Latest directory on e00: {e00_latest}\n")

    # Step 2: Create local collection directory
    local_collection_dir = LOCAL_BASE_DIR / e00_latest
    try:
        local_collection_dir.mkdir(parents=True, exist_ok=True)
        print("Step 2: Created local collection directory")
        print(f"  Path: {local_collection_dir}\n")
    except Exception as e:
        print(f"Error creating directory: {e}")
        sys.exit(1)

    # Step 2.5: Check which Pis have already been copied
    already_copied = get_already_copied_pis(local_collection_dir)
    if already_copied:
        print("Step 2.5: Found existing data")
        print(f"  Already copied from: {', '.join(sorted(already_copied))}")
        print("  Will check completeness and copy missing files\n")

    # Step 3: Copy from all Pis in parallel
    MAX_WORKERS = 8
    print(f"Step 3: Copying data from {len(PI_NAMES)} Pis "
          f"(up to {MAX_WORKERS} in parallel)...")
    print("-" * 70)

    successful = 0
    failed = 0
    skipped = 0
    partial = 0

    def process_pi(pi_name):
        """Process a single Pi: check, sync, or copy."""
        pi_remote_dir = get_remote_data_dir(pi_name)

        if pi_name in already_copied:
            local_pi_dir = None
            for item in local_collection_dir.iterdir():
                if item.is_dir() and item.name.startswith(f"{pi_name}_"):
                    local_pi_dir = item
                    break

            if local_pi_dir:
                remote_dir_name = local_pi_dir.name.split("_", 1)[1]
                is_complete, remote_count, local_count = is_copy_complete(
                    pi_name, remote_dir_name, pi_remote_dir, local_pi_dir)

                if is_complete:
                    print(f"Skipping {pi_name}: All {local_count} "
                          f"PNG files copied ✓")
                    return ("skipped", pi_name)
                print(f"Syncing {pi_name}: Remote has {remote_count} files, "
                      f"local has {local_count}")
                if copy_missing_files_from_pi(
                        pi_name, remote_dir_name, pi_remote_dir, local_pi_dir):
                    return ("partial", pi_name)
                return ("failed", pi_name)

        latest_dir = get_latest_directory(pi_name, pi_remote_dir)
        if latest_dir:
            if copy_data_from_pi(pi_name, latest_dir, pi_remote_dir,
                                 local_collection_dir):
                return ("success", pi_name)
            return ("failed", pi_name)
        return ("failed", pi_name)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_pi, p): p for p in PI_NAMES}
        for future in as_completed(futures):
            pi_name = futures[future]
            try:
                status, _ = future.result()
                if status == "skipped":
                    skipped += 1
                elif status == "partial":
                    successful += 1
                    partial += 1
                elif status == "success":
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ✗ Unexpected error for {pi_name}: {e}")
                failed += 1

    # Step 4: Summary
    print("-" * 70)
    print("\nStep 4: Summary")
    print(f"  Total Pis: {len(PI_NAMES)}")
    print(f"  Skipped (complete): {skipped}")
    print(f"  Synced (incomplete): {partial}")
    print(f"  New copies: {successful - partial}")
    print(f"  Total successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Collection directory: {local_collection_dir}")
    print("\n" + "=" * 70)

    if failed == 0:
        print("✓ All data successfully processed!")
    else:
        print(f"⚠ {failed} Pi(s) failed to copy data")
    print("=" * 70)


if __name__ == "__main__":
    main()
