import pandas as pd
import re
import pickle
import os
import time
import subprocess
from pathlib import Path
from tqdm import tqdm
import tqdm.notebook as tqdm_notebook
from ollama import Client
from concurrent.futures import ThreadPoolExecutor, as_completed
client = Client()

tqdm.pandas()


################################################
# CHECKPOINTING (for both leakage and scoring) #
################################################

def save_checkpoint(checkpoint_file, results, processed_indices, metadata=None):
    """
    Save current progress to checkpoint file.

    Args:
        checkpoint_file: Path to save checkpoint file
        results: List of processed results
        processed_indices: Set/list of processed indices
        metadata: Optional additional data to save

    Returns: bool indicating success
    """

    checkpoint_data = {
        'results': results,
        'processed_indices': list(processed_indices) if isinstance(processed_indices, set) else processed_indices,
        'metadata': metadata or {}
    }

    # Save to temporary file first to avoid corruption
    temp_file = f"{checkpoint_file}.tmp"
    try:
        with open(temp_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        # Rename temp file to actual checkpoint file (atomic operation on Unix)
        os.replace(temp_file, checkpoint_file)
        #print(f"\nCheckpoint saved: {len(results)} items processed")
        return True
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False

def load_checkpoint(checkpoint_file):
    """
    Load existing checkpoint if available.

    Args:
        checkpoint_file: Path to checkpoint file

    Returns:
        tuple: (results, processed_indices, metadata) or (None, None, None) if no checkpoint
    """
    import pickle
    import os

    if not os.path.exists(checkpoint_file):
        return None, None, None

    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            results = checkpoint_data.get('results', [])
            processed_indices = set(checkpoint_data.get('processed_indices', []))
            metadata = checkpoint_data.get('metadata', {})
            print(f"Loaded checkpoint with {len(results)} processed items")
            return results, processed_indices, metadata
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, None

def remove_ids_from_checkpoint(checkpoint_file: str, ids_to_remove: list):
    """
    Remove specific IDs from a checkpoint file.
    
    Args:
        checkpoint_file: Path to the checkpoint file
        ids_to_remove: list of IDs to remove
    """
    
    # Convert single ID to list
    if isinstance(ids_to_remove, str):
        ids_to_remove = [ids_to_remove]
    ids_to_remove = set(ids_to_remove)
    
    # Load checkpoint
    with open(checkpoint_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get components
    results = data.get('results', [])
    processed_ids = set(data.get('processed_indices', []))
    metadata = data.get('metadata', {})
    
    # Count before
    before_results = len(results)
    before_processed = len(processed_ids)
    
    # Filter results
    results = [r for r in results if r.get('id') not in ids_to_remove]
    
    # Update processed_ids
    processed_ids = processed_ids - ids_to_remove
    
    # Save
    new_data = {
        'results': results,
        'processed_indices': list(processed_ids),
        'metadata': metadata
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(new_data, f)
    
    # Print summary
    print(f"Removed {before_results - len(results)} items from results")
    print(f"Removed {before_processed - len(processed_ids)} IDs from processed_indices")
    print(f"Checkpoint updated: {checkpoint_file}")


###########
# GENERAL #
###########

def ollama_alive():
    try:
        client.list()   # lightweight call
        return True
    except:
        return False

def restart_ollama(wait=8):
    """Kill and restart Ollama to free memory/KV cache. Works on macOS."""
    print("\nRestarting Ollama...")
    subprocess.run(["pkill", "-x", "ollama"], capture_output=True)
    time.sleep(3)
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + wait
    while time.time() < deadline:
        if ollama_alive():
            print("Ollama restarted successfully.")
            return True
        time.sleep(1)
    print("Warning: Ollama did not come back up in time.")
    return False

def prompt(prompt, model="mistral:7b-instruct-v0.3-q3_K_L", temp=1.2, seed=4, num_predict=500,
            top_k=40, top_p=0.9):
    if prompt is None or pd.isna(prompt) or str(prompt).lower() == 'nan':
        return None
    else:
        response = client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": temp,            # High variation
                "seed": seed,            # Different seed each time
                "num_predict": num_predict,
                "top_k": top_k,                 # More token variety
                "top_p": top_p                  # Higher nucleus sampling
            }
        )
        return response['response']

def parse_score(text, score_code):
    """
    Parse a single score from text based on the given score code.

    Args:
        text: String containing the text to parse
        score_code: The code to look for (e.g., 'MELD', 'Na', 'Cr')

    Returns:
        float or None: The parsed score if found, None otherwise
    """
    if not text or not isinstance(text, str) or not score_code:
        return None

    pattern = rf"""
        (?<!\w)               # not preceded by letter/number
        \*{{0,2}}             # optional **
        {re.escape(score_code)}  # exact code
        [\*\s:\-<>#'\+*/()=]*   # allow weird separators of arbitrary length
        (\d+(?:\.\d+)?)       # capture score, allow decimals
        (?!\d)                # not followed by digit
    """

    match = re.search(pattern, text, re.IGNORECASE | re.VERBOSE)

    if match:
        return float(match.group(1))

    return None

def parse_all_scores(text, score_columns):
    """
    Parse text to find lines containing score codes and return:
      - the entire matching line
      - the extracted numeric score (via parse_score)

    Args:
        text: The response text to parse
        score_columns: List of score codes to look for

    Returns:
        Dictionary:
            {
                score_code: {
                    "line": full matching line,
                    "score": numeric score (float) or None
                }
            }
    """
    if not text or not isinstance(text, str):
        return {}

    lines = text.strip().split('\n')
    results = {}

    for code in score_columns:
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            score = parse_score(stripped, code)
            if score is not None:
                results[code] = {
                    "line": stripped,
                    "score": score
                }
                break

        # Fallback: code exists but no numeric score parsed
        if code not in results:
            for line in lines:
                stripped = line.strip()
                if re.search(rf'(?<!\w){re.escape(code)}',
                             stripped, re.IGNORECASE):
                    results[code] = {
                        "line": stripped,
                        "score": None
                    }
                    break

    return results


###########
# LEAKAGE #
###########

def split_into_sentences(text):
    # Simple regex-based sentence splitter
    sentences = re.split(r'(?<=[.!?])\s*|\s?-+\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def filter_zero_sentences(text, generate_leak_flag_func, model="mistral:7b-instruct-v0.3-q3_K_L"):
    """
    Iterate over sentences, call generate_leak_flag,
    keep only sentences with info: 0.
    Returns a list of dicts with sentence and parsed scores.
    """
    if text is None or pd.isna(text) or str(text).lower() == 'nan':
        return []

    results = []
    sentences = split_into_sentences(text)

    for sent in sentences:
        response = generate_leak_flag_func(sent, model=model)

        # Check if info is 0 (using the new parse_score function)
        info_score = parse_score(response, "info")

        # Only keep sentences with info: 0
        if info_score == 0:
            # For consistency, still return a scores dict with the info value
            scores = {"info": info_score}

            results.append({
                "sentence": sent,
                "scores": scores,
                "response": response
            })

    return results

def get_zero_info_sentences(text, generate_leak_flag_func, model="mistral:7b-instruct-v0.3-q3_K_L", score_columns=["info"]):
    """
    Returns a tuple: (clean_text, removed_sentences, removed_count)
    - clean_text: string with only sentences where scores['info'] == 0
    - removed_sentences: list of dicts, each with 'index', 'sentence', and 'response'
    - removed_count: integer count of removed sentences
    """
    # Process ALL sentences first
    if text is None or pd.isna(text) or str(text).lower() == 'nan':
        return "", [], 0

    sentences = split_into_sentences(text)
    kept = []
    removed = []

    for i, sent in enumerate(sentences):
        response = generate_leak_flag_func(sent, model=model)
        info_score = parse_score(response, "info")

        if info_score == 0:
            kept.append(sent)
        else:
            removed.append({
                "index": i,
                "sentence": sent,
                "response": response
            })

    clean_text = " ".join(kept)
    removed_count = len(removed)

    return clean_text, removed, removed_count

def _process_row_clean(idx, row, generate_leak_flag_func, model, text_column="description", clean_column="clean"):
    try:
        text_value = row[text_column]
        
        clean, removed, count = get_zero_info_sentences(
            text_value,
            generate_leak_flag_func,
            model=model
        )
    except Exception as e:
        print(f"Ollama call failed for ID {row.get('id')}: {e}")
        return None  # critical for not appending empty results

    # If model returned nothing meaningful, also skip
    if clean is None:
        return None

    return {
        'index': idx,
        'id': row.get('id', None),
        'name': row.get('name', None),
        text_column: text_value,
        clean_column: clean,
        'removed': removed,
        'removed_count': count
    }

def clean_df(df, generate_leak_flag_func, model="mistral:7b-instruct-v0.3-q3_K_L", workers=8,
             checkpoint_file="data/dr/scoring/clean_text.pkl", save_frequency=10, restart_every=200,
             text_column="description", clean_column="clean"):
    """
    Process startup descriptions and return a scored DataFrame with checkpointing.

    Args:
        df: Input DataFrame with at least 'description' column.
            It may also contain 'id' and 'name' columns.
        model: Ollama model name.
        workers: Number of parallel workers. 1 = sequential.
        checkpoint_file: Path to save checkpoint file.
        save_frequency: How often to save checkpoints (in number of processed rows).

    Returns:
        DataFrame with columns:
            ['id', 'name', 'description', 'clean', 'removed', 'removed_count']
    """

    # Check if Ollama is running
    if not ollama_alive():
        raise RuntimeError("Ollama is not running. Aborting.")

    # Create directory for checkpoint file if it doesn't exist
    checkpoint_dir = os.path.dirname(checkpoint_file)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created directory: {checkpoint_dir}")

    results = []
    processed_ids = set()

    # Load existing checkpoint if available
    if os.path.exists(checkpoint_file):
        loaded_results, loaded_ids, _ = load_checkpoint(checkpoint_file)
        if loaded_results is not None:
            results = loaded_results
            processed_ids = loaded_ids

    # Filter out already processed rows based on 'id' column
    remaining_df = df[~df['id'].isin(processed_ids)]
    print(f"Processing {len(remaining_df)} remaining rows out of {len(df)} total")

    # Define save_checkpoint as a nested function that captures results and processed_ids
    def save_checkpoint_callback():
        if len(results) == 0:
            return False
        return save_checkpoint(checkpoint_file, results, processed_ids)

    if workers <= 1:
        # Sequential processing
        for idx, (original_idx, row) in enumerate(tqdm(remaining_df.iterrows(), total=len(remaining_df))):
            result = _process_row_clean(original_idx, row, generate_leak_flag_func, model, text_column, clean_column)
            if result is not None: # only append valid results
                results.append(result)
                processed_ids.add(row['id'])

            # Save checkpoint periodically
            if (idx + 1) % save_frequency == 0:
                save_checkpoint_callback()
    else:
        # Batched parallel processing — never holds more than batch_size futures in memory
        batch_size = workers * 4
        rows_list = list(remaining_df.iterrows())
        total = len(rows_list)
        completed = 0

        with tqdm(total=total) as pbar:
            for batch_start in range(0, total, batch_size):
                batch = rows_list[batch_start: batch_start + batch_size]

                futures = {}
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for original_idx, row in batch:
                        future = executor.submit(_process_row_clean, original_idx, row, generate_leak_flag_func, model, text_column, clean_column)
                        futures[future] = row['id']

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result is not None:
                                results.append(result)
                                processed_ids.add(futures[future])

                        except Exception as e:
                            print(f"Error processing row with ID {futures[future]}: {e}")

                        completed += 1
                        pbar.update(1)

                # After each batch: save checkpoint and restart Ollama
                save_checkpoint_callback()
                if restart_every and completed % restart_every == 0:
                    restart_ollama()


    # Final save
    save_checkpoint_callback()

    # Build DataFrame from list of dicts
    dr_scores = pd.DataFrame(results)

    # Ensure all original rows are present (fill missing with NaN)
    dr_scores = dr_scores.reindex(df.index)

    # Select final columns in desired order
    final_cols = ['id', 'name', text_column, clean_column, 'removed', 'removed_count']

    return dr_scores[final_cols]

import spacy
nlp = spacy.load("en_core_web_sm")

def combine_masks(text):
    """
    Combine consecutive identical masks ONLY when they're truly consecutive
    with only separators between them.
    """
    pattern = r'(\[[^\]]+\])(?:(?:\s*,\s*|\s+and\s+|\s+)?\1)+'

    while re.search(pattern, text):
        text = re.sub(pattern, r'\1', text)

    return text


def spacy_mask(text, names=None, investor_names=None):
    # Mask 'names' (full + tokens)
    if names is None:
        name_list = []
    elif isinstance(names, list):
        name_list = names
    else:
        name_list = [names]

    for name in name_list:
        if pd.notna(name) and str(name).lower() != "nan":
            name = str(name).strip()

            # Mask full name (case-insensitive)
            full_pattern = r'\b' + re.escape(name) + r'\b'
            text = re.sub(full_pattern, "[NAME]", text, flags=re.IGNORECASE)

            # Mask individual tokens
            tokens = re.split(r'[.\s]+', name)
            tokens = [t for t in tokens if t]
            for token in tokens:
                token_pattern = r'\b' + re.escape(token) + r'\b'
                text = re.sub(token_pattern, "[NAME]", text, flags=re.IGNORECASE)


    # Mask 'investor_names' (full match only)
    if investor_names is None:
        investor_list = []
    elif isinstance(investor_names, list):
        investor_list = investor_names
    else:
        investor_list = [investor_names]

    for inv in investor_list:
        if pd.notna(inv) and str(inv).lower() != "nan":
            inv = str(inv).strip()

            # Only mask full name, no individual tokens
            full_pattern = r'\b' + re.escape(inv) + r'\b'
            text = re.sub(full_pattern, "[NAME]", text, flags=re.IGNORECASE)

    # SpaCy masking
    doc = nlp(text)
    label_map = {
        "PERSON": "[NAME]",
        "NORP": "[NAME]",
        "PRODUCT": "[NAME]",
        "GPE": "[LOCATION]",
        "LOC": "[LOCATION]",
        "FAC": "[LOCATION]",
        "LANGUAGE": "[LANGUAGE]",
        "DATE": "[DATE]",
        "EVENT": "[EVENT]",
        "MONEY": "[NUMBER]",
        "PERCENT": "[NUMBER]",
    }

    for ent in reversed(doc.ents):
        if ent.label_ in label_map:
            replacement = label_map[ent.label_]
            text = text[:ent.start_char] + replacement + text[ent.end_char:]

    text = combine_masks(text)
    return text


###########
# SCORING #
###########

def _process_row_score(idx, row, text_col, score_dict, generate_scores_func, model, generate_kwargs=None):
    """
    Process a single row safely.
    Returns None if model fails.
    """
    generate_kwargs = generate_kwargs or {}
    
    try:
        result = generate_scores_func(
            row[text_col],
            score_dict,
            model=model,
            **generate_kwargs
        )
    except Exception as e:
        print(f"Call failed for ID {row.get('id')}: {e}")
        return None

    # If model returned nothing meaningful → skip
    if result is None or not isinstance(result, dict):
        print(f"Invalid score result for ID {row.get('id')}")
        return None

    parsed = {
        'index': idx,
        'id': row['id'],
        'name': row['name'],
        text_col: row[text_col],
    }

    # Validate all score entries exist
    for code in score_dict.keys():
        if code not in result or result[code] is None:
            print(f"Incomplete score for ID {row.get('id')} (missing {code})")
            return None

        parsed[f'{code}_score'] = result[code]['score']
        parsed[f'{code}_response'] = result[code]['response']

    return parsed

def score_df(df, score_dict, text_col, generate_scores_func, generate_kwargs=None,
             model="mistral:7b-instruct-v0.3-q3_K_L",
             workers=8,
             checkpoint_file="data/dr/scoring/scores_checkpoint.pkl",
             save_frequency=10,
             restart_every=200):
    """
    Process startup descriptions and return scored dataframe with checkpointing.
    Uses batched parallel processing to control memory and Ollama load.
    """

    # Check if Ollama is running
    if not ollama_alive():
        raise RuntimeError("Ollama is not running. Aborting scoring.")

    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.dirname(checkpoint_file)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created directory: {checkpoint_dir}")

    results = []
    processed_ids = set()

    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        loaded_results, loaded_ids, _ = load_checkpoint(checkpoint_file)
        if loaded_results is not None:
            results = loaded_results
            processed_ids = loaded_ids

    # Filter already processed
    remaining_df = df[~df['id'].isin(processed_ids)]
    print(f"Processing {len(remaining_df)} remaining rows out of {len(df)} total")

    # Early exit if nothing left
    if len(remaining_df) == 0:
        print("All rows already processed. Loading from checkpoint.")
        dr_scores = pd.DataFrame(results)

        base_cols = ['id', 'name', text_col]
        score_cols = [f'{code}_score' for code in score_dict.keys()]
        response_cols = [f'{code}_response' for code in score_dict.keys()]
        all_cols = base_cols + [col for col in (score_cols + response_cols) if col in dr_scores.columns]

        dr_scores = dr_scores.reindex(df.index)
        return dr_scores[all_cols]

    # Nested checkpoint saver
    def save_checkpoint_callback():
        if len(results) == 0:
            return False
        return save_checkpoint(checkpoint_file, results, processed_ids)

    if workers <= 1:
        # Sequential mode (unchanged behavior)
        for idx, (original_idx, row) in enumerate(
                tqdm(remaining_df.iterrows(), total=len(remaining_df))):

            try:
                result = _process_row_score(
                    original_idx, row, text_col,
                    score_dict, generate_scores_func,
                    model, generate_kwargs
                )

                if result is not None:
                    results.append(result)
                    processed_ids.add(row['id'])

                if (idx + 1) % save_frequency == 0:
                    save_checkpoint_callback()

            except Exception as e:
                print(f"Error processing row with ID {row['id']}: {e}")

    else:
        # Batched parallel processing
        batch_size = workers * 4
        rows_list = list(remaining_df.iterrows())
        total = len(rows_list)
        completed = 0

        with tqdm(total=total) as pbar:
            for batch_start in range(0, total, batch_size):

                batch = rows_list[batch_start: batch_start + batch_size]
                futures = {}

                # Create fresh thread pool per batch
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for original_idx, row in batch:
                        future = executor.submit(
                            _process_row_score,
                            original_idx,
                            row,
                            text_col,
                            score_dict,
                            generate_scores_func,
                            model,
                            generate_kwargs
                        )
                        futures[future] = row['id']

                    # Process as completed
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result is not None:
                                results.append(result)
                                processed_ids.add(futures[future])

                        except Exception as e:
                            print(f"Error processing row with ID {futures[future]}: {e}")

                        completed += 1
                        pbar.update(1)

                # Save checkpoint after each batch
                save_checkpoint_callback()

                # Optional controlled restart
                if restart_every and completed % restart_every == 0:
                    restart_ollama()

    # Final save
    save_checkpoint_callback()
    print(f"Final checkpoint saved with {len(results)} total rows")

    # Build DataFrame
    dr_scores = pd.DataFrame(results)

    base_cols = ['id', 'name', text_col]
    score_cols = [f'{code}_score' for code in score_dict.keys()]
    response_cols = [f'{code}_response' for code in score_dict.keys()]
    all_cols = base_cols + [col for col in (score_cols + response_cols) if col in dr_scores.columns]

    dr_scores = dr_scores.reindex(df.index)

    return dr_scores[all_cols]

def _is_empty_text(value) -> bool:
    """Return True for None, NaN, blank strings, or the literal string 'nan'."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if str(value).strip() == "" or str(value).strip().lower() == "nan":
        return True
    return False

def _process_row_score_api(idx, row, text_col, score_dict, generate_scores_func,
                       model, generate_kwargs=None):
    """
    Process one row.
 
    Empty pitch text → no API call; returns a blank result row so the ID is
    still recorded in the checkpoint and appears (with NaN scores) in the output.
 
    RuntimeError (credits exhausted) is re-raised so score_df can stop the run.
    Any other exception returns None so that single-row failures are skipped.
    """
    generate_kwargs = generate_kwargs or {}
    raw_text = row[text_col]
 
    # ── Short-circuit: empty pitch → blank row, no API call ──────────────────
    if _is_empty_text(raw_text):
        tqdm.write(f"Empty text for ID {row.get('id')} — skipping API call, appending blank row.")
        blank = {
            'index':  idx,
            'id':     row['id'],
            'name':   row['name'],
            text_col: raw_text,
        }
        for code in score_dict.keys():
            blank[f'{code}_score']    = None
            blank[f'{code}_response'] = ""
        return blank
 
    # ── Normal path: call the model ───────────────────────────────────────────
    try:
        result = generate_scores_func(
            raw_text,
            score_dict,
            model=model,
            **generate_kwargs
        )
 
    except RuntimeError:
        raise  # credits exhausted — propagate to score_df
 
    except Exception as e:
        tqdm.write(f"Call call failed for ID {row.get('id')}: {e}")
        return None  # skip this row, continue with others
 
    if result is None or not isinstance(result, dict):
        tqdm.write(f"Invalid score result for ID {row.get('id')}")
        return None
 
    parsed = {
        'index':  idx,
        'id':     row['id'],
        'name':   row['name'],
        text_col: raw_text,
    }
 
    for code in score_dict.keys():
        if code not in result or result[code] is None:
            tqdm.write(f"Incomplete score for ID {row.get('id')} (missing {code})")
            return None  # do not append partial results
        parsed[f'{code}_score']    = result[code]['score']
        parsed[f'{code}_response'] = result[code]['response']
 
    return parsed

def score_df_api(
    df,
    score_dict,
    text_col,
    generate_scores_func=None,
    generate_kwargs=None,
    model="moonshotai/kimi-k2-instruct-0905",
    workers=4,
    checkpoint_file="data/dr/scoring/scores_checkpoint.pkl",
    save_frequency=10,
    **kwargs,   # absorb restart_every and other Ollama-only kwargs silently
):
    checkpoint_dir = os.path.dirname(checkpoint_file)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created directory: {checkpoint_dir}", flush=True)

    results       = []
    processed_ids = set()

    if os.path.exists(checkpoint_file):
        loaded_results, loaded_ids, _ = load_checkpoint(checkpoint_file)
        if loaded_results is not None:
            results       = loaded_results
            processed_ids = loaded_ids

    remaining_df = df[~df['id'].isin(processed_ids)]
    print(f"Processing {len(remaining_df)} remaining rows out of {len(df)} total", flush=True)

    if len(remaining_df) == 0:
        print("All rows already processed. Loading from checkpoint.", flush=True)
        dr_scores = pd.DataFrame(results)
        base_cols = ['id', 'name', text_col]
        all_cols  = base_cols + [c for c in
                     [f'{c}_score'    for c in score_dict] +
                     [f'{c}_response' for c in score_dict]
                     if c in dr_scores.columns]
        return dr_scores.reindex(df.index)[all_cols]

    def _save():
        if results:
            save_checkpoint(checkpoint_file, results, processed_ids)

    # ── Sequential ────────────────────────────────────────────────────────────
    if workers <= 1:
        for idx, (original_idx, row) in enumerate(
                tqdm(remaining_df.iterrows(), total=len(remaining_df))):
            try:
                result = _process_row_score_api(
                    original_idx, row, text_col,
                    score_dict, generate_scores_func, model, generate_kwargs
                )

            except RuntimeError as e:
                # Credits exhausted — save progress and stop
                print(f"\n*** {e} ***", flush=True)
                _save()
                raise

            except Exception as e:
                print(f"Unexpected error for row {row['id']}: {e}", flush=True)
                continue  # skip row, keep going

            # Only append if result is fully valid
            if result is not None:
                results.append(result)
                processed_ids.add(row['id'])

            if (idx + 1) % save_frequency == 0:
                _save()

    # ── Parallel ──────────────────────────────────────────────────────────────
    else:
        batch_size = workers * 4
        rows_list  = list(remaining_df.iterrows())
        total      = len(rows_list)

        with tqdm(total=total) as pbar:
            for batch_start in range(0, total, batch_size):
                batch   = rows_list[batch_start: batch_start + batch_size]
                futures = {}

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for original_idx, row in batch:
                        future = executor.submit(
                            _process_row_score_api,
                            original_idx, row, text_col,
                            score_dict, generate_scores_func, model, generate_kwargs
                        )
                        futures[future] = row['id']

                    for future in as_completed(futures):
                        try:
                            result = future.result()

                        except RuntimeError as e:
                            # Credits exhausted — save progress, cancel remaining, stop
                            tqdm.write(f"\n*** {e} ***", file=__import__('sys').stderr)
                            _save()
                            for f in futures:
                                f.cancel()
                            raise

                        except Exception as e:
                            tqdm.write(f"Unexpected error for row {futures[future]}: {e}")
                            pbar.update(1)
                            continue  # skip row, keep going

                        # Only append if result is fully valid
                        if result is not None:
                            results.append(result)
                            processed_ids.add(futures[future])

                        pbar.update(1)

                _save()

    _save()
    print(f"Final checkpoint saved with {len(results)} total rows", flush=True)

    dr_scores = pd.DataFrame(results)
    base_cols = ['id', 'name', text_col]
    all_cols  = base_cols + [c for c in
                 [f'{c}_score'    for c in score_dict] +
                 [f'{c}_response' for c in score_dict]
                 if c in dr_scores.columns]

    return dr_scores.reindex(df.index)[all_cols]