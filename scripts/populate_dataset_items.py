"""
Script to populate dataset_items table from CSV files.

Each CSV file should have columns: Elementary, Intermediate, Advanced
All rows in each column are joined into one paragraph per column.
Each CSV file creates one row in dataset_items table.
"""
import csv
import os
import re
import sys
from io import StringIO
from pathlib import Path
from uuid import UUID

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.session import SessionLocal
from app.models.dataset import DatasetItem, Dataset

# Configuration
DATASET_ID = UUID("7709a94b-1d9e-4240-a141-31b69bd90c44")
CSV_DIRECTORY = Path("/Users/shtilmanilan/Downloads/Texts-Together-OneCSVperFile")


def clean_text(text: str) -> str:
    """
    Clean text artifacts before insertion.
    - Removes replacement characters ()
    - Normalizes whitespace (multiple spaces/newlines to single space)
    - Trims edges
    """
    if not text:
        return text
    
    # Remove replacement characters ()
    text = text.replace('', '')
    
    # Normalize whitespace: multiple spaces/newlines/tabs to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Trim edges
    text = text.strip()
    
    return text


def join_column_values(values: list) -> str:
    """Join a list of values into a single paragraph, filtering out empty values."""
    # Filter out None, empty strings, and whitespace-only strings
    non_empty = [str(v).strip() for v in values if v and str(v).strip()]
    # Join with a space to create a paragraph
    joined = " ".join(non_empty)
    # Clean the joined text
    return clean_text(joined)


def process_csv_file(csv_path: Path) -> dict:
    """
    Process a single CSV file and return the joined text for each column.
    Tries multiple encodings to handle different file formats.
    
    Returns:
        dict with keys: text_ele, text_int, text_adv
    """
    elementary_texts = []
    intermediate_texts = []
    advanced_texts = []
    
    # Try multiple encodings in order of likelihood
    encodings = ['utf-8', 'windows-1252', 'iso-8859-1', 'utf-8-sig', 'cp1252']
    
    for encoding in encodings:
        try:
            # Read file in binary mode first to remove NUL bytes, then decode
            with open(csv_path, 'rb') as f:
                content = f.read()
                # Remove NUL bytes (\x00)
                content = content.replace(b'\x00', b'')
                # Decode to string
                text_content = content.decode(encoding)
            
            # Parse CSV from string
            reader = csv.DictReader(StringIO(text_content))
            
            # Normalize column names (strip whitespace) to handle variations
            for row in reader:
                # Create normalized row with stripped keys
                normalized_row = {k.strip(): v for k, v in row.items()}
                
                # Extract values from each column (using normalized names)
                if 'Elementary' in normalized_row:
                    elementary_texts.append(normalized_row['Elementary'])
                if 'Intermediate' in normalized_row:
                    intermediate_texts.append(normalized_row['Intermediate'])
                if 'Advanced' in normalized_row:
                    advanced_texts.append(normalized_row['Advanced'])
            
            # If we got here, the encoding worked
            return {
                'text_ele': join_column_values(elementary_texts),
                'text_int': join_column_values(intermediate_texts),
                'text_adv': join_column_values(advanced_texts)
            }
        except UnicodeDecodeError:
            # Try next encoding
            continue
        except Exception as e:
            # Other error, log and try next encoding
            if encoding == encodings[-1]:
                # Last encoding failed, give up
                print(f"Error processing {csv_path.name}: {e}")
                return None
            continue
    
    # If all encodings failed
    print(f"Error processing {csv_path.name}: Could not decode with any encoding")
    return None


def populate_dataset_items():
    """Main function to populate dataset_items from all CSV files."""
    db = SessionLocal()
    
    try:
        # Verify dataset exists
        dataset = db.query(Dataset).filter(Dataset.dataset_id == DATASET_ID).first()
        if not dataset:
            print(f"Error: Dataset with ID {DATASET_ID} not found in database")
            return
        print(f"✓ Found dataset: {dataset.dataset_name} (ID: {DATASET_ID})")
        
        # Get all CSV files
        csv_files = list(CSV_DIRECTORY.glob("*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {CSV_DIRECTORY}")
            return
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Process each CSV file
        success_count = 0
        error_count = 0
        
        for csv_file in csv_files:
            print(f"Processing: {csv_file.name}")
            
            # Process the CSV file
            data = process_csv_file(csv_file)
            
            if not data:
                error_count += 1
                continue
            
            # Check if text_adv is not empty (required field)
            if not data['text_adv']:
                print(f"  Warning: {csv_file.name} has no Advanced text, skipping")
                error_count += 1
                continue
            
            # Validate alignment quality
            validation_warnings = []
            validation_errors = []
            
            # 1. Check if all three levels exist
            has_all_three = bool(data['text_adv'] and data['text_int'] and data['text_ele'])
            if not has_all_three:
                missing = []
                if not data['text_adv']: missing.append('Advanced')
                if not data['text_int']: missing.append('Intermediate')
                if not data['text_ele']: missing.append('Elementary')
                validation_warnings.append(f"Missing levels: {', '.join(missing)}")
            
            # 2. Minimum length requirements
            MIN_LENGTH = 50  # Minimum characters for a valid text
            if data['text_adv'] and len(data['text_adv']) < MIN_LENGTH:
                validation_warnings.append(f"Advanced text too short ({len(data['text_adv'])} chars, min {MIN_LENGTH})")
            if data['text_int'] and len(data['text_int']) < MIN_LENGTH:
                validation_warnings.append(f"Intermediate text too short ({len(data['text_int'])} chars, min {MIN_LENGTH})")
            if data['text_ele'] and len(data['text_ele']) < MIN_LENGTH:
                validation_warnings.append(f"Elementary text too short ({len(data['text_ele'])} chars, min {MIN_LENGTH})")
            
            # 3. Check for suspicious patterns (identical or very similar texts)
            if data['text_adv'] and data['text_ele']:
                # Check if texts are identical (after normalization)
                adv_normalized = data['text_adv'].lower().strip()
                ele_normalized = data['text_ele'].lower().strip()
                if adv_normalized == ele_normalized:
                    validation_errors.append("Advanced and Elementary texts are identical")
                # Check similarity (if more than 95% similar, might be an issue)
                elif len(adv_normalized) > 0 and len(ele_normalized) > 0:
                    # Simple similarity check: count common words
                    adv_words = set(adv_normalized.split())
                    ele_words = set(ele_normalized.split())
                    if len(adv_words) > 0 and len(ele_words) > 0:
                        similarity = len(adv_words & ele_words) / max(len(adv_words), len(ele_words))
                        if similarity > 0.95:
                            validation_warnings.append(f"Very high similarity between Advanced and Elementary ({similarity:.1%})")
            
            # 4. Length progression checks
            if data['text_adv'] and data['text_ele']:
                len_adv = len(data['text_adv'])
                len_ele = len(data['text_ele'])
                # Elementary should generally be shorter, but allow some flexibility
                if len_ele > len_adv * 1.3:
                    validation_warnings.append(f"Elementary ({len_ele} chars) significantly longer than Advanced ({len_adv} chars)")
                # Advanced should generally be longer
                elif len_adv < len_ele * 0.8:
                    validation_warnings.append(f"Advanced ({len_adv} chars) shorter than Elementary ({len_ele} chars)")
            
            if data['text_int'] and data['text_adv']:
                len_int = len(data['text_int'])
                len_adv = len(data['text_adv'])
                # Intermediate should generally be between Elementary and Advanced
                if len_int > len_adv * 1.2:
                    validation_warnings.append(f"Intermediate ({len_int} chars) longer than Advanced ({len_adv} chars)")
            
            # 5. Word count and complexity checks
            def count_words(text):
                return len([w for w in text.split() if w.strip()])
            
            def avg_word_length(text):
                words = [w.strip() for w in text.split() if w.strip()]
                if not words:
                    return 0
                return sum(len(w) for w in words) / len(words)
            
            if data['text_adv'] and data['text_ele']:
                adv_words = count_words(data['text_adv'])
                ele_words = count_words(data['text_ele'])
                adv_avg_len = avg_word_length(data['text_adv'])
                ele_avg_len = avg_word_length(data['text_ele'])
                
                # Advanced should generally have more words or longer words
                if ele_words > adv_words * 1.2:
                    validation_warnings.append(f"Elementary has more words ({ele_words}) than Advanced ({adv_words})")
                
                # Advanced should generally have longer average word length (more complex vocabulary)
                if ele_avg_len > adv_avg_len * 1.1:
                    validation_warnings.append(f"Elementary avg word length ({ele_avg_len:.1f}) similar/longer than Advanced ({adv_avg_len:.1f})")
            
            # 6. Check for encoding issues (suspicious characters)
            suspicious_chars = ['', '', '', '', '', '', '', '']
            for level_name, text in [('Advanced', data['text_adv']), 
                                     ('Intermediate', data['text_int']), 
                                     ('Elementary', data['text_ele'])]:
                if text:
                    found_suspicious = [char for char in suspicious_chars if char in text]
                    if found_suspicious:
                        validation_warnings.append(f"{level_name} contains suspicious encoding characters")
            
            # 7. Check for empty or whitespace-only texts (shouldn't happen after cleaning, but double-check)
            if data['text_adv'] and not data['text_adv'].strip():
                validation_errors.append("Advanced text is empty after cleaning")
            if data['text_int'] and not data['text_int'].strip():
                validation_warnings.append("Intermediate text is empty after cleaning")
            if data['text_ele'] and not data['text_ele'].strip():
                validation_warnings.append("Elementary text is empty after cleaning")
            
            # Log validation issues
            if validation_errors:
                print(f"Validation ERRORS: {'; '.join(validation_errors)}")
                # Decide: should we skip items with errors? For now, just warn
            if validation_warnings:
                print(f"Validation warnings: {'; '.join(validation_warnings)}")
            
            # Create DatasetItem (text is already cleaned in join_column_values)
            dataset_item = DatasetItem(
                dataset_id=DATASET_ID,
                text_adv=data['text_adv'],
                text_int=data['text_int'] if data['text_int'] else None,
                text_ele=data['text_ele'] if data['text_ele'] else None
            )
            
            db.add(dataset_item)
            success_count += 1
            print(f"  ✓ Created item (Advanced: {len(data['text_adv'])} chars, "
                  f"Intermediate: {len(data['text_int'])} chars, "
                  f"Elementary: {len(data['text_ele'])} chars)")
        
        # Commit all changes
        db.commit()
        print(f"\n✓ Successfully created {success_count} dataset items")
        if error_count > 0:
            print(f"⚠ {error_count} files had errors or were skipped")
        
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    populate_dataset_items()

