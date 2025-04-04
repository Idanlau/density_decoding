#!/usr/bin/env python
"""
Script to list available ProbeInsertions, select a valid PID,
download required electrophysiology datasets, and print instructions
to run the decoding pipeline.
"""

import os
from one.api import ONE
from pathlib import Path

def list_insertions():
    one = ONE()
    # Query Alyx for all available insertions
    insertions = one.alyx.rest('insertions', 'list')
    insertions_list = list(insertions)
    print("Available insertions:")
    for ins in insertions_list:
        print(f"PID: {ins['id']}  |  Probe Name: {ins.get('name','N/A')}  |  Session: {ins.get('session', 'N/A')}")
    return insertions_list

def download_datasets(eid, local_dir, datasets):
    """
    Download the specified datasets for a given session (eid)
    into local_dir using ONE.
    """
    one = ONE()
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    for dset in datasets:
        print(f"Downloading dataset '{dset}' for session {eid}...")
        # This function downloads the dataset if needed and returns its local path.
        local_file = one.load_dataset(eid, dset)
        print(f"--> Downloaded to: {local_file}")
    return

if __name__ == '__main__':
    # 1. List available insertions
    insertions = list_insertions()
    if not insertions:
        print("No insertions found. Exiting.")
        exit(1)

    # 2. Select a valid PID; here we simply take the first one.
    selected_insertion = insertions[0]
    pid = selected_insertion['id']
    # Extract the session id (eid) from the insertion record.
    # In ONE, the insertion record usually has a 'session' field.
    eid = selected_insertion.get('session')
    if not eid:
        print("No session id (eid) found in the selected insertion. Exiting.")
        exit(1)
    print(f"\nSelected PID: {pid}")
    print(f"Corresponding EID: {eid}\n")

    # 3. Specify a local directory to download ephys data.
    # Change this to a location where you want your data stored.
    ephys_dir = "./ephys_data"
    
    # List of datasets to download. Modify as needed.
    required_datasets = [
        "spike_index.npy",
        "localization_results.npy"
    ]
    
    download_datasets(eid, ephys_dir, required_datasets)
    
    # 4. Print instructions to run the decoding pipeline.
    # You also need to provide an output path for your results.
    out_path = "./ephys_output"  # change as needed
    
    print("\nData download complete!")
    print("You can now run the decoding pipeline with a command such as:")
    print(f"singularity exec --nv --overlay /scratch/yl9727/neuro_env/overlay-15GB-500K.ext3:ro \\")
    print(f"  /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \\")
    print(f"  /bin/bash -c \"source /ext3/env.sh; python main.py --pid {pid} --ephys_path {ephys_dir} --out_path {out_path}\"")
