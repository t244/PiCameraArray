#!/usr/bin/env python3
"""
Data gathering script for PiCameraArray.
Collects data from all 16 Raspberry Pis (e00 to e15) via scp.
"""

import subprocess
import os
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
PI_NAMES = [f"e{i:02d}" for i in range(16)]  # e00 to e15
REMOTE_DATA_DIR_SSD = "/media/pi/HIKSEMI/data"  # External SSD mount point
REMOTE_DATA_DIR_LOCAL = "/home/pi/PiCameraArray/data"  # Local SD card fallback


def find_base_directory():
    """
    Find the base directory for storing collected data.
    Uses local PiCameraArray/collected_data directory.
    
    Returns:
        Path to the base directory
    """
    workspace_root = Path.cwd()
    local_path = workspace_root / "collected_data"
    print(f"Local collection directory: {local_path}")
    return local_path


LOCAL_BASE_DIR = find_base_directory()


def run_command(cmd, check=True):
    """
    Execute a shell command and return the result.
    
    Args:
        cmd: Command to run (list or string)
        check: If True, raise exception on non-zero return code
        
    Returns:
        CompletedProcess object
    """
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if check and result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Error: {result.stderr}")
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
    
    return result


def get_remote_data_dir(pi_name):
    """
    Determine which data directory exists on the Pi (SSD or local).
    
    Args:
        pi_name: Name of the Pi (e.g., 'e00')
        
    Returns:
        Path to the data directory on the Pi, or None if neither exists
    """
    try:
        # Try SSD first
        cmd = [
            "sshpass",
            "-p",
            "pi",
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            f"pi@{pi_name}.local",
            f"test -d {REMOTE_DATA_DIR_SSD} && echo 'ssd' || echo 'local'"
        ]
        
        result = run_command(cmd, check=True)
        location = result.stdout.strip()
        
        if location == "ssd":
            return REMOTE_DATA_DIR_SSD
        else:
            return REMOTE_DATA_DIR_LOCAL
            
    except Exception as e:
        print(f"  Warning: Could not determine data dir on {pi_name}, using SD card: {e}")
        return REMOTE_DATA_DIR_LOCAL


def get_latest_directory(pi_name, remote_data_dir):
    """
    Get the latest data directory from a Pi.
    
    Args:
        pi_name: Name of the Pi (e.g., 'e00')
        remote_data_dir: Path to the data directory on the Pi
        
    Returns:
        Directory name (string) or None if not found
    """
    try:
        cmd = [
            "sshpass",
            "-p",
            "pi",
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            f"pi@{pi_name}.local",
            f"ls -t {remote_data_dir} | head -1"
        ]
        
        result = run_command(cmd, check=True)
        latest_dir = result.stdout.strip()
        
        if latest_dir:
            return latest_dir
        else:
            print(f"Warning: No data directory found on {pi_name} at {remote_data_dir}")
            return None
            
    except Exception as e:
        print(f"Error getting latest directory from {pi_name}: {e}")
        return None


def copy_data_from_pi(pi_name, remote_dir_name, remote_data_dir, local_dir):
    """
    Copy data directory from a Pi using rsync.

    Args:
        pi_name: Name of the Pi (e.g., 'e00')
        remote_dir_name: Name of the remote directory to copy
        remote_data_dir: Path to the data directory on the Pi
        local_dir: Local destination directory

    Returns:
        True if successful, False otherwise
    """
    try:
        remote_path = f"pi@{pi_name}.local:{remote_data_dir}/{remote_dir_name}/"
        local_path = local_dir / f"{pi_name}_{remote_dir_name}"
        local_path.mkdir(parents=True, exist_ok=True)

        print(f"Copying from {pi_name}... ({remote_dir_name})")

        cmd = [
            "sshpass",
            "-p",
            "pi",
            "rsync",
            "-av",
            "-e",
            "ssh -o StrictHostKeyChecking=no",
            remote_path,
            str(local_path) + "/"
        ]

        result = run_command(cmd, check=True)
        print(f"  ✓ Successfully copied to {local_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error copying from {pi_name}: {e}")
        return False


def copy_missing_files_from_pi(pi_name, remote_dir_name, remote_data_dir, local_dir_path):
    """
    Copy only missing PNG files from a Pi using rsync.
    
    Args:
        pi_name: Name of the Pi (e.g., 'e00')
        remote_dir_name: Name of the remote directory
        remote_data_dir: Path to the data directory on the Pi
        local_dir_path: Path to the local directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        remote_path = f"pi@{pi_name}.local:{remote_data_dir}/{remote_dir_name}/"
        
        print(f"Syncing missing files from {pi_name}... ({remote_dir_name})")
        
        cmd = [
            "sshpass",
            "-p",
            "pi",
            "rsync",
            "-av",
            "--ignore-existing",
            "-e",
            'ssh -o StrictHostKeyChecking=no',
            remote_path,
            str(local_dir_path) + "/"
        ]
        
        result = run_command(cmd, check=True)
        print(f"  ✓ Successfully synced missing files to {local_dir_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error syncing missing files from {pi_name}: {e}")
        return False


def get_already_copied_pis(local_dir):
    """
    Check which Pis have already had their data copied.
    
    Args:
        local_dir: Local collection directory path
        
    Returns:
        Set of Pi names that have already been copied (e.g., {'e00', 'e01'})
    """
    already_copied = set()
    
    if not local_dir.exists():
        return already_copied
    
    for item in local_dir.iterdir():
        if item.is_dir():
            # Extract Pi name from directory name (e.g., 'e00_2024-01-01_120000')
            dir_name = item.name
            if '_' in dir_name:
                pi_name = dir_name.split('_')[0]
                if pi_name in PI_NAMES:
                    already_copied.add(pi_name)
    
    return already_copied


def count_remote_png_files(pi_name, remote_dir_name, remote_data_dir):
    """
    Count PNG files in the remote directory on a Pi.
    
    Args:
        pi_name: Name of the Pi (e.g., 'e00')
        remote_dir_name: Name of the remote directory
        remote_data_dir: Path to the data directory on the Pi
        
    Returns:
        Number of PNG files or -1 if error
    """
    try:
        cmd = [
            "sshpass",
            "-p",
            "pi",
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            f"pi@{pi_name}.local",
            f"find {remote_data_dir}/{remote_dir_name} -name '*.png' | wc -l"
        ]
        
        result = run_command(cmd, check=True)
        count = int(result.stdout.strip())
        return count
        
    except Exception as e:
        print(f"  Error counting remote PNG files on {pi_name}: {e}")
        return -1


def count_local_png_files(local_dir_path):
    """
    Count PNG files in a local directory recursively.
    
    Args:
        local_dir_path: Path to the local directory
        
    Returns:
        Number of PNG files
    """
    if not local_dir_path.exists():
        return 0
    
    count = 0
    for png_file in local_dir_path.rglob("*.png"):
        count += 1
    
    return count


def is_copy_complete(pi_name, remote_dir_name, remote_data_dir, local_dir_path):
    """
    Check if all PNG files have been copied from a Pi.
    
    Args:
        pi_name: Name of the Pi (e.g., 'e00')
        remote_dir_name: Name of the remote directory
        remote_data_dir: Path to the data directory on the Pi
        local_dir_path: Path to the local directory
        
    Returns:
        Tuple (is_complete: bool, remote_count: int, local_count: int)
    """
    remote_count = count_remote_png_files(pi_name, remote_dir_name, remote_data_dir)
    local_count = count_local_png_files(local_dir_path)
    
    if remote_count < 0:
        return False, remote_count, local_count
    
    return remote_count == local_count, remote_count, local_count


def main():
    """Main function to orchestrate data gathering."""
    
    print("=" * 70)
    print("PiCameraArray Data Gathering Script")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base directory: {LOCAL_BASE_DIR}\n")
    
    # Check if sshpass is installed
    try:
        result = subprocess.run(["sshpass", "-V"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error: sshpass is not installed or not in PATH")
            print("Please install sshpass to use this script without SSH key prompts")
            sys.exit(1)
    except FileNotFoundError:
        print("Error: sshpass is not installed or not in PATH")
        print("Please install sshpass:")
        print("  - On Windows: choco install sshpass (requires Chocolatey)")
        print("  - Or use: scoop install sshpass")
        sys.exit(1)
    
    # Check if rsync is installed
    try:
        result = subprocess.run(["rsync", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error: rsync is not installed or not in PATH")
            print("Please install rsync to sync missing files")
            sys.exit(1)
    except FileNotFoundError:
        print("Error: rsync is not installed or not in PATH")
        print("Please install rsync:")
        print("  - On Windows: choco install rsync (requires Chocolatey)")
        print("  - Or use: scoop install rsync")
        sys.exit(1)
    
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
        print(f"Step 2: Created local collection directory")
        print(f"  Path: {local_collection_dir}\n")
    except Exception as e:
        print(f"Error creating directory: {e}")
        sys.exit(1)
    
    # Step 2.5: Check which Pis have already been copied
    already_copied = get_already_copied_pis(local_collection_dir)
    
    if already_copied:
        print(f"Step 2.5: Found existing data")
        print(f"  Already copied from: {', '.join(sorted(already_copied))}")
        print(f"  Will check completeness and copy missing files\n")
    
    # Step 3: Retrieve the latest directory name from e00 for all Pis (should be the same)
    # Get latest directory info from e00 to compare with other Pis
    MAX_WORKERS = 8
    print(f"Step 3: Copying data from {len(PI_NAMES)} Pis (up to {MAX_WORKERS} in parallel)...")
    print("-" * 70)

    successful = 0
    failed = 0
    skipped = 0
    partial = 0

    def process_pi(pi_name):
        """Process a single Pi: check, sync, or copy. Returns (status, pi_name)."""
        # Determine data directory for this Pi
        pi_remote_dir = get_remote_data_dir(pi_name)

        # Check if this Pi's data already exists locally
        if pi_name in already_copied:
            local_pi_dir = None
            # Find the directory for this Pi
            for item in local_collection_dir.iterdir():
                if item.is_dir() and item.name.startswith(f"{pi_name}_"):
                    local_pi_dir = item
                    break

            if local_pi_dir:
                # Extract the remote directory name from the local directory
                remote_dir_name = local_pi_dir.name.split('_', 1)[1]

                # Check if copy is complete
                is_complete, remote_count, local_count = is_copy_complete(
                    pi_name, remote_dir_name, pi_remote_dir, local_pi_dir
                )

                if is_complete:
                    print(f"Skipping {pi_name}: All {local_count} PNG files copied ✓")
                    return ("skipped", pi_name)
                else:
                    print(f"Syncing {pi_name}: Remote has {remote_count} files, local has {local_count}")
                    if copy_missing_files_from_pi(pi_name, remote_dir_name, pi_remote_dir, local_pi_dir):
                        return ("partial", pi_name)
                    else:
                        return ("failed", pi_name)

        # Get latest directory for this Pi (first time copy)
        latest_dir = get_latest_directory(pi_name, pi_remote_dir)

        if latest_dir:
            if copy_data_from_pi(pi_name, latest_dir, pi_remote_dir, local_collection_dir):
                return ("success", pi_name)
            else:
                return ("failed", pi_name)
        else:
            return ("failed", pi_name)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_pi, pi_name): pi_name for pi_name in PI_NAMES}

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
    print(f"\nStep 4: Summary")
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
