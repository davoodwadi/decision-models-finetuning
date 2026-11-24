#!/usr/bin/env python3
"""
Generate:
1. question_catalog.json - All unique questions across wave1_3_persona_json, wave4_Q_wave1_3_A, wave4_Q_wave4_A
2. wave1_3_response.csv - Answers from wave1_3_persona_json + wave4_Q_wave1_3_A, columns indexed by QuestionID
3. wave4_response.csv - Answers from wave4_Q_wave4_A, columns indexed by QuestionID
"""

import json
import os
import sys
from pathlib import Path
from collections import OrderedDict
import pandas as pd

BASE_DIR = Path(__file__).parent / 'hf_dataset'

DIRS = {
    'wave1_3_persona_json': BASE_DIR / 'wave1_3_persona_json',
    'wave4_Q_wave1_3_A': BASE_DIR / 'wave4_Q_wave1_3_A',
    'wave4_Q_wave4_A': BASE_DIR / 'wave4_Q_wave4_A',
}


def get_all_pid_files(directory: Path):
    """Get all pid_*.json files sorted by pid number."""
    files = sorted(directory.glob('pid_*.json'), key=lambda f: int(f.stem.split('_')[1]))
    return files


def extract_questions_from_json(json_path: Path):
    """Extract all questions (without answers) from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        blocks = json.load(f)

    questions = []
    if not isinstance(blocks, list):
        blocks = [blocks]

    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_name = block.get('BlockName', '')
        block_id = block.get('BlockID', '')
        block_questions = block.get('Questions', [])
        for q in block_questions:
            q_copy = dict(q)
            q_copy['_BlockName'] = block_name
            q_copy['_BlockID'] = block_id
            questions.append(q_copy)

    return questions


def build_question_catalog(dirs: dict):
    """Scan all JSON files across all directories, collect all unique QuestionIDs with metadata."""
    catalog = OrderedDict()  # QuestionID -> metadata

    for dir_name, dir_path in dirs.items():
        files = get_all_pid_files(dir_path)
        if not files:
            print(f"Warning: No files found in {dir_path}")
            continue

        print(f"Scanning {len(files)} files in {dir_name}...")

        for json_path in files:
            questions = extract_questions_from_json(json_path)
            for q in questions:
                qid = q.get('QuestionID', '')
                if not qid:
                    continue

                # Build a unique catalog key: use QID + BlockName to disambiguate
                # questions that share the same QID but appear in different blocks
                # with different types
                block_name = q.get('_BlockName', '')
                catalog_key = qid
                if qid in catalog:
                    existing = catalog[qid]
                    if existing['QuestionType'] != q.get('QuestionType', '') or \
                       existing['BlockName'] != block_name:
                        # Duplicate QID with different type/block — add both
                        catalog_key = f"{qid}___{block_name}"
                        if catalog_key in catalog:
                            continue  # Already stored this variant

                if catalog_key not in catalog:
                    # Build metadata entry (exclude Answers)
                    entry = OrderedDict()
                    entry['QuestionID'] = qid
                    entry['QuestionText'] = q.get('QuestionText', '')
                    entry['QuestionType'] = q.get('QuestionType', '')

                    if q.get('is_descriptive'):
                        entry['is_descriptive'] = True

                    # Options (for MC)
                    if 'Options' in q:
                        entry['Options'] = q['Options']

                    # Rows/RowsID (for Matrix)
                    if 'Rows' in q:
                        entry['Rows'] = q['Rows']
                    if 'RowsID' in q:
                        entry['RowsID'] = q['RowsID']

                    # Columns (for Matrix)
                    if 'Columns' in q:
                        entry['Columns'] = q['Columns']

                    # Statements/StatementsID (for Slider)
                    if 'Statements' in q:
                        entry['Statements'] = q['Statements']
                    if 'StatementsID' in q:
                        entry['StatementsID'] = q['StatementsID']

                    # Range (for Slider)
                    if 'Range' in q:
                        entry['Range'] = q['Range']

                    # Settings
                    if 'Settings' in q:
                        entry['Settings'] = q['Settings']

                    entry['BlockName'] = block_name
                    entry['source'] = dir_name

                    # Determine CSV column names this question expands to
                    entry['csv_columns'] = get_csv_columns(q)

                    catalog[catalog_key] = entry

    return catalog


def get_csv_columns(question: dict):
    """Determine what CSV column names a question expands to."""
    qid = question.get('QuestionID', '')
    qtype = question.get('QuestionType', '')
    settings = question.get('Settings', {})

    if question.get('is_descriptive') or qtype == 'DB':
        return []  # Descriptive blocks produce no columns

    if qtype == 'MC':
        selector = settings.get('Selector', '')
        if selector in ['MAVR', 'MAHR']:
            # Multi-select: expand to one binary column per option (matching wave1-3 anonymized format)
            options = question.get('Options', [])
            return [f"{qid}_{i+1}" for i in range(len(options))]
        else:
            return [qid]

    elif qtype == 'Matrix':
        # Expand by RowsID or StatementsID
        if 'StatementsID' in question:
            ids = question['StatementsID']
        elif 'RowsID' in question:
            ids = question['RowsID']
        elif 'Statements' in question:
            ids = [str(i + 1) for i in range(len(question['Statements']))]
        elif 'Rows' in question:
            ids = [str(i + 1) for i in range(len(question['Rows']))]
        else:
            return [qid]
        return [f"{qid}_{rid}" for rid in ids]

    elif qtype == 'Slider':
        if 'StatementsID' in question:
            ids = question['StatementsID']
        elif 'Statements' in question:
            stmts = question['Statements']
            if len(stmts) > 1:
                ids = [str(i + 1) for i in range(len(stmts))]
            else:
                return [qid]
        else:
            return [qid]
        return [f"{qid}_{sid}" for sid in ids]

    elif qtype == 'TE':
        selector = settings.get('Selector', '')
        if selector == 'FORM':
            if 'RowsID' in question:
                ids = question['RowsID']
            elif 'Rows' in question:
                ids = [str(i + 1) for i in range(len(question['Rows']))]
            else:
                return [f"{qid}_TEXT"]
            return [f"{qid}_{rid}" for rid in ids]
        else:
            return [f"{qid}_TEXT"]

    else:
        return [qid]


def extract_answers_from_json(json_path: Path):
    """Extract answers from a JSON file, returning dict of column_name -> value."""
    with open(json_path, 'r', encoding='utf-8') as f:
        blocks = json.load(f)

    answers = {}

    if not isinstance(blocks, list):
        blocks = [blocks]

    for block in blocks:
        if not isinstance(block, dict):
            continue

        questions = block.get('Questions', [])

        for question in questions:
            qid = question.get('QuestionID', '')
            qtype = question.get('QuestionType', '')
            settings = question.get('Settings', {})
            answer_data = question.get('Answers', {})

            if not answer_data or not qid:
                continue

            if question.get('is_descriptive') or qtype == 'DB':
                continue

            if qtype == 'MC':
                selector = settings.get('Selector', '')
                if selector in ['SAVR', 'SAHR']:
                    pos = answer_data.get('SelectedByPosition', '')
                    if pos is not None and pos != '':
                        try:
                            answers[qid] = int(pos)
                        except (ValueError, TypeError):
                            answers[qid] = answer_data.get('SelectedText', '')
                    else:
                        answers[qid] = answer_data.get('SelectedText', '')
                else:
                    # Multi-select (MAVR/MAHR): expand to binary columns per option
                    # matching wave1-3 anonymized format: {QID}_{choiceId} = 1 if selected
                    options = question.get('Options', [])
                    num_options = len(options)
                    positions = answer_data.get('SelectedByPosition', [])
                    if isinstance(positions, list) and positions:
                        selected_set = set(int(p) for p in positions)
                    else:
                        selected_set = set()
                    for i in range(num_options):
                        col = f"{qid}_{i+1}"
                        answers[col] = 1 if (i + 1) in selected_set else None

            elif qtype == 'Matrix':
                selected = answer_data.get('SelectedByPosition', [])

                if 'StatementsID' in question:
                    ids = question['StatementsID']
                elif 'RowsID' in question:
                    ids = question['RowsID']
                elif 'Statements' in question:
                    ids = [str(i + 1) for i in range(len(question['Statements']))]
                elif 'Rows' in question:
                    ids = [str(i + 1) for i in range(len(question['Rows']))]
                else:
                    ids = []

                for i, rid in enumerate(ids):
                    if selected and i < len(selected):
                        val = selected[i]
                        try:
                            answers[f"{qid}_{rid}"] = int(val)
                        except (ValueError, TypeError):
                            answers[f"{qid}_{rid}"] = val
                    else:
                        answers[f"{qid}_{rid}"] = None

            elif qtype == 'Slider':
                values = answer_data.get('Values', [])

                if 'StatementsID' in question:
                    ids = question['StatementsID']
                elif 'Statements' in question:
                    stmts = question['Statements']
                    if len(stmts) > 1:
                        ids = [str(i + 1) for i in range(len(stmts))]
                    else:
                        ids = None
                else:
                    ids = None

                if ids:
                    for i, sid in enumerate(ids):
                        if i < len(values):
                            v = values[i]
                            try:
                                answers[f"{qid}_{sid}"] = float(v) if v is not None else None
                            except (ValueError, TypeError):
                                answers[f"{qid}_{sid}"] = v
                        else:
                            answers[f"{qid}_{sid}"] = None
                else:
                    if values:
                        v = values[0]
                        try:
                            answers[qid] = float(v) if v is not None else None
                        except (ValueError, TypeError):
                            answers[qid] = v

            elif qtype == 'TE':
                selector = settings.get('Selector', '')
                if selector == 'FORM':
                    texts = answer_data.get('Text', [])
                    if 'RowsID' in question:
                        ids = question['RowsID']
                    elif 'Rows' in question:
                        ids = [str(i + 1) for i in range(len(question['Rows']))]
                    else:
                        ids = None

                    if ids and isinstance(texts, list):
                        for i, text_dict in enumerate(texts):
                            if i < len(ids):
                                rid = ids[i]
                                for _, value in text_dict.items():
                                    answers[f"{qid}_{rid}"] = str(value) if value else None
                    else:
                        text = answer_data.get('Text', '')
                        answers[f"{qid}_TEXT"] = str(text) if text else None
                else:
                    text = answer_data.get('Text', '')
                    answers[f"{qid}_TEXT"] = str(text) if text else None

            else:
                # Fallback
                if 'Values' in answer_data:
                    values = answer_data['Values']
                    if isinstance(values, list) and len(values) == 1:
                        answers[qid] = values[0]
                    else:
                        answers[qid] = str(values)
                elif 'Text' in answer_data:
                    answers[qid] = str(answer_data['Text'])

    return answers


def extract_label_answers_from_json(json_path: Path):
    """Extract text/label answers from a JSON file, returning dict of column_name -> text value."""
    with open(json_path, 'r', encoding='utf-8') as f:
        blocks = json.load(f)

    answers = {}

    if not isinstance(blocks, list):
        blocks = [blocks]

    for block in blocks:
        if not isinstance(block, dict):
            continue

        questions = block.get('Questions', [])

        for question in questions:
            qid = question.get('QuestionID', '')
            qtype = question.get('QuestionType', '')
            settings = question.get('Settings', {})
            answer_data = question.get('Answers', {})

            if not answer_data or not qid:
                continue

            if question.get('is_descriptive') or qtype == 'DB':
                continue

            if qtype == 'MC':
                selector = settings.get('Selector', '')
                if selector in ['SAVR', 'SAHR']:
                    text = answer_data.get('SelectedText', '')
                    if text:
                        answers[qid] = text
                    else:
                        # Fall back: try to map position to option text
                        pos = answer_data.get('SelectedByPosition', '')
                        options = question.get('Options', [])
                        if pos is not None and pos != '':
                            try:
                                idx = int(pos) - 1
                                if 0 <= idx < len(options):
                                    answers[qid] = options[idx]
                            except (ValueError, TypeError):
                                pass
                else:
                    # Multi-select (MAVR/MAHR): expand to binary columns per option
                    # matching wave1-3 anonymized format: {QID}_{choiceId} = option text if selected
                    options = question.get('Options', [])
                    num_options = len(options)
                    positions = answer_data.get('SelectedByPosition', [])
                    if isinstance(positions, list) and positions:
                        selected_set = set(int(p) for p in positions)
                    else:
                        selected_set = set()
                    for i in range(num_options):
                        col = f"{qid}_{i+1}"
                        answers[col] = options[i] if (i + 1) in selected_set else None

            elif qtype == 'Matrix':
                selected_text = answer_data.get('SelectedText', [])
                selected_pos = answer_data.get('SelectedByPosition', [])
                columns = question.get('Columns', [])

                if 'StatementsID' in question:
                    ids = question['StatementsID']
                elif 'RowsID' in question:
                    ids = question['RowsID']
                elif 'Statements' in question:
                    ids = [str(i + 1) for i in range(len(question['Statements']))]
                elif 'Rows' in question:
                    ids = [str(i + 1) for i in range(len(question['Rows']))]
                else:
                    ids = []

                for i, rid in enumerate(ids):
                    if selected_text and i < len(selected_text):
                        answers[f"{qid}_{rid}"] = selected_text[i]
                    elif selected_pos and i < len(selected_pos) and columns:
                        try:
                            idx = int(selected_pos[i]) - 1
                            if 0 <= idx < len(columns):
                                answers[f"{qid}_{rid}"] = columns[idx]
                        except (ValueError, TypeError):
                            answers[f"{qid}_{rid}"] = None
                    else:
                        answers[f"{qid}_{rid}"] = None

            elif qtype == 'Slider':
                # Sliders are numeric by nature, use values as strings
                values = answer_data.get('Values', [])

                if 'StatementsID' in question:
                    ids = question['StatementsID']
                elif 'Statements' in question:
                    stmts = question['Statements']
                    if len(stmts) > 1:
                        ids = [str(i + 1) for i in range(len(stmts))]
                    else:
                        ids = None
                else:
                    ids = None

                if ids:
                    for i, sid in enumerate(ids):
                        if i < len(values):
                            answers[f"{qid}_{sid}"] = str(values[i]) if values[i] is not None else None
                        else:
                            answers[f"{qid}_{sid}"] = None
                else:
                    if values:
                        answers[qid] = str(values[0]) if values[0] is not None else None

            elif qtype == 'TE':
                selector = settings.get('Selector', '')
                if selector == 'FORM':
                    texts = answer_data.get('Text', [])
                    if 'RowsID' in question:
                        ids = question['RowsID']
                    elif 'Rows' in question:
                        ids = [str(i + 1) for i in range(len(question['Rows']))]
                    else:
                        ids = None

                    if ids and isinstance(texts, list):
                        for i, text_dict in enumerate(texts):
                            if i < len(ids):
                                rid = ids[i]
                                for _, value in text_dict.items():
                                    answers[f"{qid}_{rid}"] = str(value) if value else None
                    else:
                        text = answer_data.get('Text', '')
                        answers[f"{qid}_TEXT"] = str(text) if text else None
                else:
                    text = answer_data.get('Text', '')
                    answers[f"{qid}_TEXT"] = str(text) if text else None

            else:
                if 'Text' in answer_data:
                    answers[qid] = str(answer_data['Text'])
                elif 'Values' in answer_data:
                    values = answer_data['Values']
                    if isinstance(values, list) and len(values) == 1:
                        answers[qid] = str(values[0])
                    else:
                        answers[qid] = str(values)

    return answers


def extract_pid(filepath: Path):
    """Extract pid number from filename like pid_3.json."""
    return int(filepath.stem.split('_')[1])


def generate_csvs():
    """Generate numeric and label CSVs for wave1_3 and wave4."""

    # Get file lists
    persona_files = get_all_pid_files(DIRS['wave1_3_persona_json'])
    wave1_3_a_files = get_all_pid_files(DIRS['wave4_Q_wave1_3_A'])
    wave4_a_files = get_all_pid_files(DIRS['wave4_Q_wave4_A'])

    # Build pid -> filepath mappings
    persona_by_pid = {extract_pid(f): f for f in persona_files}
    wave1_3_a_by_pid = {extract_pid(f): f for f in wave1_3_a_files}
    wave4_a_by_pid = {extract_pid(f): f for f in wave4_a_files}

    # All pids across persona + wave1_3_A
    all_pids_wave1_3 = sorted(set(persona_by_pid.keys()) | set(wave1_3_a_by_pid.keys()))
    all_pids_wave4 = sorted(wave4_a_by_pid.keys())

    # --- wave1_3 ---
    print(f"\nGenerating wave1_3 CSVs for {len(all_pids_wave1_3)} participants...")
    wave1_3_rows = []
    wave1_3_label_rows = []
    for i, pid in enumerate(all_pids_wave1_3):
        row = {'pid': pid}
        label_row = {'pid': pid}

        if pid in persona_by_pid:
            row.update(extract_answers_from_json(persona_by_pid[pid]))
            label_row.update(extract_label_answers_from_json(persona_by_pid[pid]))

        if pid in wave1_3_a_by_pid:
            row.update(extract_answers_from_json(wave1_3_a_by_pid[pid]))
            label_row.update(extract_label_answers_from_json(wave1_3_a_by_pid[pid]))

        wave1_3_rows.append(row)
        wave1_3_label_rows.append(label_row)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(all_pids_wave1_3)}...")

    for rows, name in [(wave1_3_rows, 'wave1_3_response.csv'),
                        (wave1_3_label_rows, 'wave1_3_response_label.csv')]:
        df = pd.DataFrame(rows)
        cols = ['pid'] + [c for c in df.columns if c != 'pid']
        df = df[cols]
        path = BASE_DIR / name
        df.to_csv(path, index=False)
        print(f"Saved {name}: {df.shape[0]} rows x {df.shape[1]} columns")

    # --- wave4 ---
    print(f"\nGenerating wave4 CSVs for {len(all_pids_wave4)} participants...")
    wave4_rows = []
    wave4_label_rows = []
    for i, pid in enumerate(all_pids_wave4):
        row = {'pid': pid}
        label_row = {'pid': pid}

        row.update(extract_answers_from_json(wave4_a_by_pid[pid]))
        label_row.update(extract_label_answers_from_json(wave4_a_by_pid[pid]))

        wave4_rows.append(row)
        wave4_label_rows.append(label_row)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(all_pids_wave4)}...")

    for rows, name in [(wave4_rows, 'wave4_response.csv'),
                        (wave4_label_rows, 'wave4_response_label.csv')]:
        df = pd.DataFrame(rows)
        cols = ['pid'] + [c for c in df.columns if c != 'pid']
        df = df[cols]
        path = BASE_DIR / name
        df.to_csv(path, index=False)
        print(f"Saved {name}: {df.shape[0]} rows x {df.shape[1]} columns")


def main():
    print("=" * 60)
    print("Step 1: Building question catalog...")
    print("=" * 60)

    catalog = build_question_catalog(DIRS)

    catalog_path = BASE_DIR / 'question_catalog.json'
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(list(catalog.values()), f, indent=2, ensure_ascii=False)

    print(f"\nSaved question_catalog.json with {len(catalog)} unique questions")

    # Print summary
    type_counts = {}
    for entry in catalog.values():
        qt = entry['QuestionType']
        type_counts[qt] = type_counts.get(qt, 0) + 1
    print(f"Question types: {type_counts}")

    total_cols = sum(len(entry['csv_columns']) for entry in catalog.values())
    print(f"Total CSV columns (excluding pid): {total_cols}")

    print("\n" + "=" * 60)
    print("Step 2: Generating CSVs (numeric + label)...")
    print("=" * 60)

    generate_csvs()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
