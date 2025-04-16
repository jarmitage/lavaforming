#!/usr/bin/env python3
"""
Logic Pro Marker Reader and Writer

A comprehensive tool for working with Logic Pro marker data in audio files. This script provides
functionality to:

1. Extract Markers:
   - Read marker positions and labels from Logic Pro audio files
   - Support for multiple audio formats through soundfile library
   - Fallback to raw WAV chunk analysis when needed
   - Handle both sample-based and SMPTE timecode formats

2. Export/Import:
   - Save markers to CSV files for easy editing
   - Import markers from CSV files
   - Convert between sample positions and SMPTE timecode

3. Write Markers:
   - Write marker data back to audio files
   - Preserve original audio quality while adding marker metadata
   - Support for cue points and labels in WAV format

Usage:
    # Display help and usage information
    python logic_markers.py --help

    # Read markers from an audio file
    python logic_markers.py read input.wav --csv_output markers.csv

    # Write markers to an audio file
    python logic_markers.py write --input_audio input.wav --csv_file markers.csv --output_audio output.wav

    # Test the marker writing functionality
    python logic_markers.py test_write --source_audio source.wav --target_audio target.wav

Source: https://support.apple.com/en-gw/guide/logicpro/lgcpadb63ff8/mac
"""
import os
import sys
import re
from collections import OrderedDict
import struct
import fire
import csv
import shutil
import soundfile as sf
from typing import Dict, Optional, Union, Literal

class LogicMarkerBase:
    """Base class for Logic Pro marker handling with shared functionality"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.sample_rate = 48000  # Logic Pro uses 48kHz sample rate
            
    def log(self, message):
        """Print debug messages when verbose mode is enabled"""
        if self.verbose:
            print(f"{message}")
            
    def _smpte_to_samples(self, smpte_time):
        """
        Convert SMPTE timecode to sample position
        Format: HH:MM:SS:FF.SF (Hours:Minutes:Seconds:Frames.Subframes)
        
        Args:
            smpte_time (str): SMPTE timecode string (e.g., "01:00:00:12.40")
            
        Returns:
            int: Sample position
        """
        # Parse SMPTE time using regex to handle both formats with and without subframes
        match = re.match(r'(\d+):(\d+):(\d+):(\d+)(?:\.(\d+))?', smpte_time)
        if not match:
            raise ValueError(f"Invalid SMPTE timecode format: {smpte_time}")
            
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        frames = int(match.group(4))
        subframes = int(match.group(5) or 0)
        
        # Convert everything to seconds (30 fps, 100 subframes per frame)
        total_seconds = (hours * 3600) + (minutes * 60) + seconds + (frames / 30) + (subframes / 3000)
        
        # Convert to samples at specified sample rate
        return int(total_seconds * self.sample_rate)

    def _samples_to_smpte(self, position):
        """
        Convert sample position to SMPTE timecode
        
        Args:
            position (int): Sample position at 48kHz
            
        Returns:
            str: SMPTE timecode string (HH:MM:SS:FF.SF)
        """
        total_seconds = position / self.sample_rate
        hours = int(total_seconds // 3600)
        total_seconds %= 3600
        minutes = int(total_seconds // 60)
        total_seconds %= 60
        seconds = int(total_seconds)
        fractional_seconds = total_seconds - seconds
        frames = int(fractional_seconds * 30)
        subframes = int((fractional_seconds * 30 - frames) * 100)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}.{subframes:02d}"

class LogicMarkerReader(LogicMarkerBase):
    """Class to read marker information from audio files created in Logic Pro"""
    
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        
    def read_markers_from_file(self, file_path):
        """
        Read marker information from an audio file that was created in Logic Pro
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            dict: Dictionary of markers if found
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[LogicMarkerReader.read_markers_from_file] File not found: {file_path}")
            
        self.log(f"[LogicMarkerReader.read_markers_from_file] Reading markers from: {file_path}")
        
        # Try reading metadata first
        markers = {}
        self.log("[LogicMarkerReader.read_markers_from_file] Reading file metadata")
        markers = self.read_markers_from_metadata(file_path)
        if markers:
            return markers
        
        # Fall back to basic WAV chunk analysis with improved approach
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.wav':
            self.log("[LogicMarkerReader.read_markers_from_file] Using improved WAV chunk analysis")
            markers = self.read_markers_from_wav(file_path)
            if markers:
                return markers
                
        self.log("[LogicMarkerReader.read_markers_from_file] No markers found in file")
        return {}
        
    def read_markers_from_metadata(self, file_path):
        """Read markers from audio file metadata"""
        markers = {}
        try:
            with sf.SoundFile(file_path) as sound_file:
                self.log(f"[LogicMarkerReader.read_markers_from_metadata] File opened: {sound_file.format}")
                
                # Try to access metadata
                if hasattr(sound_file, 'format_info'):
                    self.log(f"[LogicMarkerReader.read_markers_from_metadata] Format info: {sound_file.format_info}")
                    
                # Check for subchunks if it's a WAV file
                if sound_file.format == 'WAV' and hasattr(sound_file, '_format') and hasattr(sound_file._format, 'subchunks'):
                    for subchunk in sound_file._format.subchunks:
                        self.log(f"[LogicMarkerReader.read_markers_from_metadata] Found subchunk: {subchunk}")
                        if 'cue' in subchunk or 'LIST' in subchunk or 'adtl' in subchunk:
                            markers[subchunk] = sound_file._format.subchunks[subchunk]
                
        except Exception as e:
            self.log(f"[LogicMarkerReader.read_markers_from_metadata] Error processing file: {e}")
            
        return markers
        
    def read_markers_from_wav(self, file_path):
        """
        Improved approach to read markers from WAV files
        Uses more robust chunk parsing focusing on Logic Pro specific metadata
        """
        markers = OrderedDict()
        
        try:
            with open(file_path, 'rb') as file:
                # Check RIFF header
                riff_header = file.read(12)
                if riff_header[0:4] != b'RIFF' or riff_header[8:12] != b'WAVE':
                    self.log("[LogicMarkerReader.read_markers_from_wav] Not a valid WAV file")
                    return markers
                    
                self.log("[LogicMarkerReader.read_markers_from_wav] Valid WAV file detected")
                
                # Read chunks
                while True:
                    try:
                        chunk_header = file.read(8)
                        if len(chunk_header) < 8:
                            break  # End of file
                            
                        chunk_id = chunk_header[0:4]
                        chunk_size = int.from_bytes(chunk_header[4:8], byteorder='little')
                        
                        # Make chunk ID readable
                        chunk_name = chunk_id.decode('utf-8', errors='replace')
                        self.log(f"[LogicMarkerReader.read_markers_from_wav] Found chunk: {chunk_name}, raw_id: {chunk_id!r}, size: {chunk_size}")
                        
                        # Process specific chunks
                        if chunk_id == b'cue ':
                            self.log("[LogicMarkerReader.read_markers_from_wav] Processing cue chunk")
                            markers.update(self._process_cue_chunk(file, chunk_size))
                        elif chunk_id == b'LIST':
                            # LIST chunk might contain marker labels in adtl subchunk
                            list_type = file.read(4)
                            if list_type == b'adtl':
                                self.log("[LogicMarkerReader.read_markers_from_wav] Processing adtl chunk")
                                markers.update(self._process_adtl_chunk(file, chunk_size - 4))
                            else:
                                # Skip the rest of chunk
                                file.seek(chunk_size - 4, 1)
                        elif chunk_id == b'bext':
                            # BWF extension chunk
                            self.log("[LogicMarkerReader.read_markers_from_wav] Processing BWF (bext) chunk")
                            markers.update(self._process_bext_chunk(file, chunk_size))
                        # Look for Apple-specific or Logic-specific chunks
                        elif chunk_name.startswith(('APPL', 'appl', 'Logi')):
                            self.log(f"[LogicMarkerReader.read_markers_from_wav] Processing potential Logic chunk: {chunk_name}")
                            chunk_data = file.read(chunk_size)
                            markers[f"{chunk_name}_data"] = chunk_data
                        else:
                            # Skip chunk
                            file.seek(chunk_size, 1)
                            
                        # Ensure reading starts on an even boundary for the next chunk
                        if file.tell() % 2 != 0:
                            self.log(f"[LogicMarkerReader.read_markers_from_wav] Aligned file pointer by skipping 1 byte after {chunk_name}")
                            file.seek(1, 1) # Skip padding byte
                            
                    except Exception as e:
                        self.log(f"[LogicMarkerReader.read_markers_from_wav] Error processing chunk: {e}")
                        break
                        
        except Exception as e:
            self.log(f"[LogicMarkerReader.read_markers_from_wav] Error reading WAV file: {e}")
            
        return markers
    
    def _process_cue_chunk(self, file, chunk_size):
        """Process a cue chunk to extract marker positions"""
        markers = {}
        try:
            # Read number of cue points
            cue_data = file.read(4)
            num_cue_points = int.from_bytes(cue_data, byteorder='little')
            self.log(f"[LogicMarkerReader._process_cue_chunk] Number of cue points: {num_cue_points}")
            
            for i in range(num_cue_points):
                # Each cue point is 24 bytes
                cue_point_data = file.read(24)
                if len(cue_point_data) < 24:
                    break
                    
                # Extract cue point data
                cue_id = int.from_bytes(cue_point_data[0:4], byteorder='little')
                position = int.from_bytes(cue_point_data[20:24], byteorder='little')
                
                # Store the position directly in samples
                markers[f"cue_{cue_id}"] = position
                
                # Convert to SMPTE for logging
                smpte_time = self._samples_to_smpte(position)
                self.log(f"[LogicMarkerReader._process_cue_chunk] Cue point {cue_id} at {smpte_time}")
                
                # Log the first cue point data for detailed check
                if i == 0:
                    self.log(f"[LogicMarkerReader._process_cue_chunk] First cue point raw data: {cue_point_data!r}")
                
        except Exception as e:
            self.log(f"[LogicMarkerReader._process_cue_chunk] Error processing cue chunk: {e}")
            
        return markers
        
    def _process_adtl_chunk(self, file, chunk_size):
        """Process an adtl (Associated Data List) chunk for marker labels"""
        markers = {}
        end_position = file.tell() + chunk_size
        self.log(f"[LogicMarkerReader._process_adtl_chunk] Entered processing adtl chunk. Reading until file position {end_position}")
        
        try:
            while file.tell() < end_position:
                # Read subchunk header
                subchunk_header = file.read(8)
                if len(subchunk_header) < 8:
                    break
                    
                subchunk_id = subchunk_header[0:4]
                subchunk_size = int.from_bytes(subchunk_header[4:8], byteorder='little')
                
                # 'labl' or 'note' subchunks contain text labels for markers
                self.log(f"[LogicMarkerReader._process_adtl_chunk] Found subchunk: {subchunk_id!r}, size: {subchunk_size}")
                if subchunk_id in (b'labl', b'note'):
                    # Read cue point ID
                    cue_id = int.from_bytes(file.read(4), byteorder='little')
                    
                    # Read label text (subchunk_size - 4 for the cue ID, and sometimes padded to even length)
                    label_length = subchunk_size - 4
                    label_data = file.read(label_length)
                    
                    # Convert to string and strip null characters
                    label = label_data.split(b'\0', 1)[0].decode('utf-8', errors='replace')
                    
                    markers[f"label_{cue_id}"] = label
                    self.log(f"[LogicMarkerReader._process_adtl_chunk] Label for cue {cue_id}: {label}")
                else:
                    # Skip other subchunks
                    self.log(f"[LogicMarkerReader._process_adtl_chunk] Skipping unknown subchunk: {subchunk_id!r}")
                    file.seek(subchunk_size, 1)
                
                # Ensure we're at an even byte boundary
                if subchunk_size % 2 != 0:
                    file.seek(1, 1)
                    
        except Exception as e:
            self.log(f"[LogicMarkerReader._process_adtl_chunk] Error processing adtl chunk: {e}")
            
        return markers
        
    def _process_bext_chunk(self, file, chunk_size):
        """
        Process a BWF extension (bext) chunk
        Logic Pro might store additional metadata here
        """
        markers = {}
        try:
            # BWF spec: https://tech.ebu.ch/docs/tech/tech3285.pdf
            # Basic fields in bext chunk
            description = file.read(256).split(b'\0', 1)[0].decode('utf-8', errors='replace')
            originator = file.read(32).split(b'\0', 1)[0].decode('utf-8', errors='replace')
            originator_ref = file.read(32).split(b'\0', 1)[0].decode('utf-8', errors='replace')
            
            # More BWF metadata fields
            date_data = file.read(10)
            time_data = file.read(8)
            
            if description:
                markers['bwf_description'] = description
            if originator:
                markers['bwf_originator'] = originator
            if originator_ref:
                markers['bwf_originator_ref'] = originator_ref
                
            # Skip the rest of the chunk
            remaining_bytes = chunk_size - 256 - 32 - 32 - 10 - 8
            if remaining_bytes > 0:
                file.seek(remaining_bytes, 1)
                
        except Exception as e:
            self.log(f"[LogicMarkerReader._process_bext_chunk] Error processing bext chunk: {e}")
            
        return markers

    def save_to_csv(self, markers: Dict[str, Union[int, str]], csv_path: str, use_smpte: bool = False):
        """
        Save markers to a CSV file in a format compatible with LogicMarkerWriter
        
        Args:
            markers (dict): Dictionary of markers
            csv_path (str): Path to save the CSV file
            use_smpte (bool): If True, save positions in SMPTE format instead of samples
        """
        # Extract cue points and their corresponding labels
        marker_data = []
        for key, value in markers.items():
            if key.startswith('cue_'):
                marker_id = key.split('_')[1]
                label = markers.get(f'label_{marker_id}', f'Marker {marker_id}')
                position = value if not use_smpte else self._samples_to_smpte(value)
                marker_data.append({
                    'position': position,
                    'label': label
                })
        
        # Sort by position (for SMPTE format, we'll sort by the original sample positions)
        if use_smpte:
            # Keep original order from sample positions
            marker_data.sort(key=lambda x: self._smpte_to_samples(x['position']))
        else:
            marker_data.sort(key=lambda x: x['position'])
        
        # Write to CSV
        if marker_data:
            with open(csv_path, 'w', newline='') as csvfile:
                # Use position_smpte as column name when in SMPTE format
                fieldnames = ['position_smpte' if use_smpte else 'position', 'label']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Rename the position key to match the fieldname if using SMPTE
                if use_smpte:
                    for data in marker_data:
                        data['position_smpte'] = data.pop('position')
                
                writer.writeheader()
                writer.writerows(marker_data)
            self.log(f"[LogicMarkerReader.save_to_csv] Saved {len(marker_data)} markers to {csv_path} in {'SMPTE' if use_smpte else 'samples'} format")
        else:
            self.log("[LogicMarkerReader.save_to_csv] No markers to save")

class LogicMarkerWriter(LogicMarkerBase):
    """Class to write marker information to audio files in Logic Pro format"""
    
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        
    def read_markers_from_csv(self, csv_path: str) -> Dict[str, Union[int, str]]:
        """
        Read markers from a CSV file
        Expected format: position,label or position_smpte,label
        
        Args:
            csv_path (str): Path to the CSV file
            
        Returns:
            dict: Dictionary of markers with positions and labels
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[LogicMarkerWriter.read_markers_from_csv] CSV file not found: {csv_path}")
            
        markers = {}
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames
            
            # Determine if we're reading SMPTE format
            is_smpte = 'position_smpte' in fieldnames if fieldnames else False
            position_field = 'position_smpte' if is_smpte else 'position'
            
            for i, row in enumerate(reader, 1):
                try:
                    position_str = row.get(position_field, '0')
                    label = row.get('label', f'Marker {i}')
                    
                    # Convert position to samples if in SMPTE format
                    if is_smpte:
                        position = self._smpte_to_samples(position_str)
                    else:
                        position = int(position_str)
                        
                    markers[f"cue_{i}"] = position
                    markers[f"label_{i}"] = label
                    if self.verbose:
                        self.log(f"[LogicMarkerWriter.read_markers_from_csv] Read marker {i}: position={position}, label={label}")
                except ValueError as e:
                    self.log(f"[LogicMarkerWriter.read_markers_from_csv] Error reading row {i}: {e}")
                    continue
                    
        if self.verbose:
            self.log(f"[LogicMarkerWriter.read_markers_from_csv] Successfully read {len(markers)//2} markers from {csv_path}")
        return markers
        
    def write_markers_to_file(self, input_audio_path: str, output_audio_path: str, markers: Dict[str, Union[int, str]]):
        """
        Write markers to a new audio file
        
        Args:
            input_audio_path (str): Path to the input audio file
            output_audio_path (str): Path to save the output audio file
            markers (dict): Dictionary of markers to write
        """
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"Audio file not found: {input_audio_path}")
            
        # First make a clean copy of the input file - don't modify it
        shutil.copy2(input_audio_path, output_audio_path)
        
        # Extract only the markers we need
        cue_markers = {k: v for k, v in markers.items() if k.startswith('cue_')}
        label_markers = {k: v for k, v in markers.items() if k.startswith('label_')}
        
        # Nothing to do if no markers
        if not cue_markers:
            self.log(f"No markers to write")
            return
        
        # Use the rewrite approach instead of append
        try:
            # We now rewrite the file in place, so the output path is the one to modify
            self._rewrite_wav_with_markers(output_audio_path, cue_markers, label_markers)
            if self.verbose:
                self.log(f"[LogicMarkerWriter.write_markers_to_file] Rewrote {output_audio_path} with {len(cue_markers)} markers.")
        except Exception as e:
            self.log(f"[LogicMarkerWriter.write_markers_to_file] Error rewriting WAV file: {e}")
            # Clean up the potentially corrupted output file if there was an error
            if os.path.exists(output_audio_path):
                os.remove(output_audio_path)
            raise
                
    def _rewrite_wav_with_markers(self, wav_path: str, cue_markers: Dict[str, int], label_markers: Dict[str, str]):
        """
        Rewrites the WAV file, replacing existing cue/label chunks with new ones.
        
        Args:
            wav_path (str): Path to the WAV file (will be overwritten)
            cue_markers (dict): Dictionary of cue markers (cue_id: position)
            label_markers (dict): Dictionary of label markers (label_id: label_text)
        """
        self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Starting rewrite for {wav_path}")
        
        # --- 1. Read all existing chunks --- 
        chunks = []
        try:
            with open(wav_path, 'rb') as f_read:
                # Read RIFF header
                riff_id = f_read.read(4)
                if riff_id != b'RIFF':
                    raise ValueError("Not a valid WAV file: missing RIFF header")
                riff_size_bytes = f_read.read(4) # Store original size bytes for now
                wave_id = f_read.read(4)
                if wave_id != b'WAVE':
                    raise ValueError("Not a valid WAV file: missing WAVE format")

                self.log("[LogicMarkerWriter._rewrite_wav_with_markers] Reading existing chunks...")
                while True:
                    chunk_header = f_read.read(8)
                    if len(chunk_header) < 8:
                        if len(chunk_header) > 0:
                            self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Warning: Trailing {len(chunk_header)} bytes found at end of file.")
                        break # End of file
                    
                    chunk_id = chunk_header[0:4]
                    chunk_size = int.from_bytes(chunk_header[4:8], byteorder='little')
                    chunk_data = f_read.read(chunk_size)

                    if len(chunk_data) < chunk_size:
                        self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Warning: Chunk {chunk_id!r} truncated. Expected {chunk_size}, got {len(chunk_data)}.")
                        # Decide how to handle - skip chunk or raise error? For now, store what we got.

                    chunks.append((chunk_id, chunk_data))
                    self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Read chunk: {chunk_id!r}, size: {chunk_size}")
                    
                    # Skip padding byte if chunk size was odd
                    if chunk_size % 2 != 0:
                        padding = f_read.read(1)
                        if not padding:
                             self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Warning: Expected padding byte after chunk {chunk_id!r} but reached EOF.")
                             break # Reached end unexpectedly
        except Exception as e:
            self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Error reading existing chunks: {e}")
            raise # Re-raise error, cannot proceed

        if not chunks:
            self.log("[LogicMarkerWriter._rewrite_wav_with_markers] No chunks found in the file. Cannot rewrite.")
            return # Or raise error
            
        # --- 2. Filter out existing marker chunks --- 
        filtered_chunks = []
        fmt_chunk_found = False
        for chunk_id, chunk_data in chunks:
            if chunk_id == b'cue ':
                self.log("[LogicMarkerWriter._rewrite_wav_with_markers] Filtering out existing cue chunk.")
                continue
            if chunk_id == b'LIST' and chunk_data.startswith(b'adtl'):
                self.log("[LogicMarkerWriter._rewrite_wav_with_markers] Filtering out existing LIST/adtl chunk.")
                continue
            if chunk_id == b'fmt ':
                 fmt_chunk_found = True
            filtered_chunks.append((chunk_id, chunk_data))
            
        if not fmt_chunk_found:
            raise ValueError("Cannot rewrite WAV file: 'fmt ' chunk is missing.")

        # --- 3. Create new marker chunks --- 
        new_cue_chunk_data = self._create_cue_chunk(cue_markers)
        new_label_chunk_data = self._create_label_chunk(cue_markers, label_markers) if label_markers else None
        
        new_marker_chunks = []
        if new_cue_chunk_data:
            new_marker_chunks.append((b'cue ', new_cue_chunk_data))
            self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Created new cue chunk, size {len(new_cue_chunk_data)}.")
        if new_label_chunk_data:
            # Note: _create_label_chunk includes the 'adtl' prefix in its data
            new_marker_chunks.append((b'LIST', new_label_chunk_data))
            self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Created new LIST/adtl chunk, size {len(new_label_chunk_data)}.")

        # --- 4. Insert new chunks (after 'fmt ') --- 
        final_chunks = []
        inserted = False
        for chunk_id, chunk_data in filtered_chunks:
            final_chunks.append((chunk_id, chunk_data))
            if chunk_id == b'fmt ' and not inserted:
                final_chunks.extend(new_marker_chunks)
                inserted = True
                self.log("[LogicMarkerWriter._rewrite_wav_with_markers] Inserted new marker chunks after 'fmt '.")
        
        if not inserted:
             # This should theoretically not happen if fmt_chunk_found was true
             self.log("[LogicMarkerWriter._rewrite_wav_with_markers] Warning: Could not find 'fmt ' chunk to insert markers after. Appending instead?")
             # Fallback or error? For safety, let's append to the end of the list.
             final_chunks.extend(new_marker_chunks)

        # --- 5. Write the new file structure --- 
        total_chunk_bytes = 0
        try:
            with open(wav_path, 'wb') as f_write:
                f_write.write(b'RIFF')
                f_write.write(b'\0\0\0\0') # Placeholder for total size
                f_write.write(b'WAVE')
                
                self.log("[LogicMarkerWriter._rewrite_wav_with_markers] Writing modified chunks...")
                for chunk_id, chunk_data in final_chunks:
                    chunk_size = len(chunk_data)
                    f_write.write(chunk_id)
                    f_write.write(struct.pack('<I', chunk_size))
                    f_write.write(chunk_data)
                    self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Wrote chunk: {chunk_id!r}, size: {chunk_size}")
                    
                    # Add padding byte if chunk data size is odd
                    if chunk_size % 2 != 0:
                        f_write.write(b'\0')
                        total_chunk_bytes += chunk_size + 1 + 8 # data + padding + id/size header
                        self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Added padding byte after {chunk_id!r}")
                    else:
                        total_chunk_bytes += chunk_size + 8 # data + id/size header
                
                # Calculate final size (total bytes written - 8 for RIFF ID/size) + 4 for WAVE ID?
                # Simpler: Get final file size using tell()
                final_size = f_write.tell()
                riff_chunk_size = final_size - 8 # The RIFF size field excludes the 'RIFF' ID and the size field itself
                
                # Go back and write the correct size
                f_write.seek(4)
                f_write.write(struct.pack('<I', riff_chunk_size))
                self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Successfully rewrote {wav_path}. Final RIFF size: {riff_chunk_size}, Total file size: {final_size}")

        except Exception as e:
            self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Error writing modified chunks: {e}")
            # Attempt to clean up potentially broken file
            if os.path.exists(wav_path):
                 os.remove(wav_path)
            raise

    def _create_cue_chunk(self, cue_markers: Dict[str, int]):
        """
        Create a properly formatted cue chunk for WAV files
        
        Args:
            cue_markers (dict): Dictionary of cue markers (cue_id: position)
            
        Returns:
            bytes: Binary data for the cue chunk
        """
        # Extract marker IDs and positions
        markers_list = []
        for key, value in cue_markers.items():
            try:
                cue_id = int(key.split('_')[1])
                markers_list.append((cue_id, value))
            except (ValueError, IndexError):
                self.log(f"Warning: Could not parse cue ID from {key}, skipping")
        
        # Sort by ID
        markers_list.sort()
        
        # Create the chunk data
        chunk_data = struct.pack('<I', len(markers_list))  # Number of cues
        
        for cue_id, position in markers_list:
            chunk_data += struct.pack('<I', cue_id)          # ID
            chunk_data += struct.pack('<I', position)        # Position
            chunk_data += struct.pack('<4s', b'data')       # Data chunk ID - properly formatted as 4 bytes
            chunk_data += struct.pack('<I', 0)               # Chunk start
            chunk_data += struct.pack('<I', 0)               # Block start
            chunk_data += struct.pack('<I', position)        # Sample offset
        
        return chunk_data
        
    def _create_label_chunk(self, cue_markers: Dict[str, int], label_markers: Dict[str, str]):
        """
        Create a properly formatted LIST/adtl chunk with labels
        
        Args:
            cue_markers (dict): Dictionary of cue markers (cue_id: position)
            label_markers (dict): Dictionary of label markers (label_id: label_text)
            
        Returns:
            bytes: Binary data for the label LIST chunk
        """
        # Start with the LIST type ("adtl")
        chunk_data = struct.pack('<4s', b'adtl')  # Properly format the adtl identifier
        
        # Extract marker IDs
        marker_ids = []
        for key in cue_markers.keys():
            try:
                cue_id = int(key.split('_')[1])
                marker_ids.append(cue_id)
            except (ValueError, IndexError):
                self.log(f"Warning: Could not parse cue ID from {key}, skipping")
        
        # Sort by ID
        marker_ids.sort()
        
        # Create label subchunks for each marker
        for cue_id in marker_ids:
            label_key = f"label_{cue_id}"
            label_text = label_markers.get(label_key, f"Marker {cue_id}")
            
            # Convert to bytes and add null terminator
            label_bytes = label_text.encode('utf-8') + b'\0'
            
            # Calculate size (4 bytes for cue ID + label length including null terminator)
            label_size = 4 + len(label_bytes)
            
            # Add label subchunk
            chunk_data += struct.pack('<4s', b'labl')      # Subchunk ID - properly formatted
            chunk_data += struct.pack('<I', label_size)    # Subchunk size
            chunk_data += struct.pack('<I', cue_id)        # Cue ID
            chunk_data += label_bytes                      # Label text with null terminator
            
            # Pad to even length if needed
            if label_size % 2 != 0:
                chunk_data += b'\0'
        
        return chunk_data

class LogicMarkerTester:
    """Class for testing Logic Pro marker reading and writing functionality"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
    def test_write(self, source_audio: str, target_audio: str):
        """
        Test the marker writing functionality by:
        1. Reading markers from a source audio file (with markers from Logic)
        2. Writing those markers to a target audio file
        3. Reading back the markers from the target file
        4. Comparing the markers to validate the write worked
        
        Args:
            source_audio: Path to audio file with markers from Logic
            target_audio: Path to audio file to write markers to
        """
        try:
            print("\n=== [LogicMarkerTester.test_write] Starting Marker Write Test ===")
            
            # Create temporary CSV file
            temp_csv = "temp_test_markers.csv"
            
            # Step 1: Read markers from source file
            print("\n1. [LogicMarkerTester.test_write] Reading markers from source file...")
            reader = LogicMarkerReader(verbose=self.verbose)
            source_markers = reader.read_markers_from_file(source_audio)
            
            if not source_markers:
                print("[LogicMarkerTester.test_write] Error: No markers found in source file!")
                return
                
            print(f"[LogicMarkerTester.test_write] Found {len([k for k in source_markers.keys() if k.startswith('cue_')])} markers in source file")
            
            # Step 2: Save markers to CSV
            print("\n2. [LogicMarkerTester.test_write] Saving markers to temporary CSV...")
            reader.save_to_csv(source_markers, temp_csv)
            
            # Step 3: Write markers to target file
            print("\n3. [LogicMarkerTester.test_write] Writing markers to target file...")
            writer = LogicMarkerWriter(verbose=self.verbose)
            csv_markers = writer.read_markers_from_csv(temp_csv)
            # Extract cue/label markers from CSV data
            csv_cue_markers = {k: v for k, v in csv_markers.items() if k.startswith('cue_')}
            csv_label_markers = {k: v for k, v in csv_markers.items() if k.startswith('label_')}

            # Use rewrite logic - copy target first, then rewrite the copy
            target_marked_path = target_audio + ".marked.wav"
            shutil.copy2(target_audio, target_marked_path)
            writer._rewrite_wav_with_markers(target_marked_path, csv_cue_markers, csv_label_markers) # Pass correct markers
            print(f"[LogicMarkerTester.test_write] Rewrote {target_marked_path} with {len(csv_cue_markers)} markers.") # OK for test info

            # Step 4: Read back markers from target file
            print("\n4. [LogicMarkerTester.test_write] Reading back markers from target file...")
            result_markers = reader.read_markers_from_file(target_marked_path)
            
            if not result_markers:
                print("[LogicMarkerTester.test_write] Error: No markers found in target file after writing!")
                return
                
            # Step 5: Compare markers
            print("\n5. [LogicMarkerTester.test_write] Comparing markers...")
            
            # Get cue points and labels from both sets
            source_cues = {k: v for k, v in source_markers.items() if k.startswith('cue_')}
            source_labels = {k: v for k, v in source_markers.items() if k.startswith('label_')}
            result_cues = {k: v for k, v in result_markers.items() if k.startswith('cue_')}
            result_labels = {k: v for k, v in result_markers.items() if k.startswith('label_')}
            
            # Compare counts
            print(f"\n[LogicMarkerTester.test_write] Source file has {len(source_cues)} markers")
            print(f"[LogicMarkerTester.test_write] Target file has {len(result_cues)} markers")
            
            # Compare positions
            print("\n[LogicMarkerTester.test_write] Comparing marker positions:")
            positions_match = True
            for cue_id in source_cues:
                if cue_id in result_cues:
                    if source_cues[cue_id] != result_cues[cue_id]:
                        print(f"Position mismatch for {cue_id}:")
                        print(f"  Source: {source_cues[cue_id]}")
                        print(f"  Target: {result_cues[cue_id]}")
                        positions_match = False
                else:
                    print(f"Missing marker in target: {cue_id}")
                    positions_match = False
            
            # Compare labels
            print("\n[LogicMarkerTester.test_write] Comparing marker labels:")
            labels_match = True
            for label_id in source_labels:
                if label_id in result_labels:
                    if source_labels[label_id] != result_labels[label_id]:
                        print(f"Label mismatch for {label_id}:")
                        print(f"  Source: {source_labels[label_id]}")
                        print(f"  Target: {result_labels[label_id]}")
                        labels_match = False
                else:
                    print(f"Missing label in target: {label_id}")
                    labels_match = False
            
            # Final result
            print("\n=== [LogicMarkerTester.test_write] Test Results ===")
            print(f"[LogicMarkerTester.test_write] Positions match: {'✓' if positions_match else '✗'}")
            print(f"[LogicMarkerTester.test_write] Labels match: {'✓' if labels_match else '✗'}")
            
            # Cleanup
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
                
        except Exception as e:
            print(f"[LogicMarkerTester.test_write] Test error: {e}")
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            sys.exit(1)

class LogicMarkerCLI:
    """Command line interface for Logic Pro marker extraction and writing"""
    
    def read(self, file_path: str, csv_output: Optional[str] = None, smpte: bool = False, 
             verbose: bool = False):
        """
        Extract marker information from Logic Pro audio files
        
        Args:
            file_path: Path to the audio file
            csv_output: Optional path to save markers as CSV
            smpte: If True, save positions in SMPTE format (default: False)
            verbose: Enable verbose output (default: False)
        """
        reader = LogicMarkerReader(verbose=verbose)
        try:
            markers = reader.read_markers_from_file(file_path)
            
            if markers:
                print("\n[LogicMarkerCLI.read] Markers found:")
                for marker_key, marker_value in markers.items():
                    # Convert to SMPTE for display if requested
                    if smpte and marker_key.startswith('cue_'):
                        marker_value = reader._samples_to_smpte(marker_value)
                    print(f"[LogicMarkerCLI.read] {marker_key}: {marker_value}")
                    
                if csv_output:
                    reader.save_to_csv(markers, csv_output, use_smpte=smpte)
                    print(f"\n[LogicMarkerCLI.read] Markers saved to CSV: {csv_output}")
            else:
                print("\n[LogicMarkerCLI.read] No markers found in the file.")
                
        except Exception as e:
            print(f"[LogicMarkerCLI.read] Error: {e}")
            sys.exit(1)
            
    def write(self, input_audio: str, csv_file: str, output_audio: str, 
              verbose: bool = False):
        """
        Write markers from a CSV file to an audio file
        
        Args:
            input_audio: Path to the input audio file
            csv_file: Path to the CSV file containing markers (format: position,label)
            output_audio: Path to save the output audio file
            verbose: Enable verbose output (default: False)
        """
        writer = LogicMarkerWriter(verbose=verbose)
        try:
            markers = writer.read_markers_from_csv(csv_file)
            writer.write_markers_to_file(input_audio, output_audio, markers)
            print(f"\n[LogicMarkerCLI.write] Successfully wrote {len(markers)//2} markers to {output_audio}")
        except Exception as e:
            print(f"[LogicMarkerCLI.write] Error: {e}")
            sys.exit(1)

    def test_write(self, source_audio: str, target_audio: str, verbose: bool = False):
        """
        Test the marker writing functionality
        
        Args:
            source_audio: Path to audio file with markers from Logic
            target_audio: Path to audio file to write markers to
            verbose: Enable verbose output (default: False)
        """
        tester = LogicMarkerTester(verbose=verbose)
        tester.test_write(source_audio, target_audio)

def main():
    fire.Fire(LogicMarkerCLI())
    
if __name__ == '__main__':
    main()