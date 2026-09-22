#!/usr/bin/env python
"""
W2-2 Builder: Large sheet test corpus (>400k cells).

Scenario: Researcher exporting 6 weeks of continuous accelerometer + vitals data
from a wearable monitor worn by 150 participants in a physical activity intervention.
Data exported at 6-hour intervals (4 readings per day).

Result: 60,900 data rows + 1 header row, 9 columns.
- Total cells: 60,901 * 9 = 548,109 (triggers large-sheet fallback in the reader)
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from openpyxl import Workbook
from openpyxl.utils import get_column_letter


def build_large_sheet(output_path: str) -> dict:
    """Build a realistic large-sheet fixture and return metadata."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Wearable Monitor Export"

    # Header row
    headers = [
        "Reading_Date",
        "Participant_ID",
        "Heart_Rate_bpm",
        "Systolic_BP_mmHg",
        "Diastolic_BP_mmHg",
        "Step_Count_hourly",
        "Activity_Type",
        "Signal_Quality",
        "Notes",
    ]
    ws.append(headers)

    # Generate 60,900 data rows
    # 150 participants × 406 readings each (6-week period, 4/day = 168 readings/week)
    n_participants = 150
    readings_per_participant = 406  # 6 weeks * 7 days * 4 readings/day = 168, but pad to 406
    total_rows = n_participants * readings_per_participant

    start_date = datetime(2025, 1, 15)
    activity_types = [
        "Rest",
        "Light Activity",
        "Moderate Activity",
        "Vigorous Activity",
    ]
    signal_quality_map = {True: 1, False: 0}  # Binary column

    row_num = 2
    for pid in range(1, n_participants + 1):
        participant_id = f"P{pid:04d}"
        current_date = start_date
        blanks_inserted = 0

        for reading_idx in range(readings_per_participant):
            # Date advances every 6 hours
            if reading_idx > 0 and reading_idx % 4 == 0:
                current_date += timedelta(days=1)

            # Heart rate: 95% of readings present, 5% intentionally blank (device dropout)
            if reading_idx % 20 == 0 and blanks_inserted < 3:  # ~3 blanks per participant
                hr = None
                blanks_inserted += 1
            else:
                # Realistic resting HR 60-100, activity HR higher
                base_hr = 65 + (pid % 20)
                activity_effect = (reading_idx % 4) * 8
                hr = int(base_hr + activity_effect + (reading_idx % 7))

            # Blood pressure
            sys_bp = 110 + (pid % 10) + (reading_idx % 20) // 2
            dia_bp = 70 + (pid % 8) + (reading_idx % 15) // 3

            # Step count: realistic hourly variation
            steps = max(0, 200 + (pid % 100) - (reading_idx % 10) * 15)

            # Activity type cycling
            activity = activity_types[reading_idx % 4]

            # Signal quality binary (0 or 1): 92% good, 8% poor
            signal_quality = signal_quality_map[reading_idx % 13 != 0]

            # Notes: mostly empty, some entries have text
            notes = ""
            if reading_idx % 50 == 0:
                notes = "Manual verification performed"
            elif reading_idx % 75 == 0:
                notes = "Device synchronized with server"

            # Write row
            ws.append(
                [
                    current_date.strftime("%Y-%m-%d %H:%M"),
                    participant_id,
                    hr,
                    sys_bp,
                    dia_bp,
                    steps,
                    activity,
                    signal_quality,
                    notes,
                ]
            )
            row_num += 1

    wb.save(output_path)
    actual_data_rows = total_rows
    actual_cols = len(headers)

    return {
        "n_data_rows": actual_data_rows,
        "n_cols": actual_cols,
        "total_cells": actual_data_rows * actual_cols,
        "header_row": 1,
        "last_data_row": 1 + actual_data_rows,
    }


def verify_file(output_path: str) -> dict:
    """Re-open the file with openpyxl read_only and verify counts."""
    from openpyxl import load_workbook

    wb = load_workbook(output_path, read_only=True, data_only=True)
    ws = wb.active

    # Count rows (including header)
    row_count = 0
    col_count = 0
    for row_idx, row in enumerate(ws.iter_rows(), start=1):
        row_count = row_idx
        if row_idx == 1:
            col_count = len([cell for cell in row if cell.value is not None])

    data_rows = row_count - 1  # Exclude header
    wb.close()

    return {
        "rows_confirmed": data_rows,
        "cols_confirmed": col_count,
        "total_cells_confirmed": data_rows * col_count,
    }


if __name__ == "__main__":
    # Build to temp path
    temp_path = "/tmp/_w2_2_big.xlsx"
    print(f"Building large sheet to {temp_path}...")
    metadata = build_large_sheet(temp_path)
    print(
        f"  Generated: {metadata['n_data_rows']} data rows × {metadata['n_cols']} cols "
        f"= {metadata['total_cells']} cells"
    )

    # Verify by reading back
    print(f"Verifying via openpyxl read_only...")
    verified = verify_file(temp_path)
    print(
        f"  Confirmed: {verified['rows_confirmed']} data rows × {verified['cols_confirmed']} cols "
        f"= {verified['total_cells_confirmed']} cells"
    )

    # Cleanup
    import os

    os.remove(temp_path)
    print(f"Temp file deleted.")
    print(f"\nVerified counts for truth.json:")
    print(f"  rows_confirmed: {verified['rows_confirmed']}")
    print(f"  cols_confirmed: {verified['cols_confirmed']}")
