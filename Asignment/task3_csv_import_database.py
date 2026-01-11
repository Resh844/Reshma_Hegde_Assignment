"""
Task 3: CSV Data Import to Database
Reads user information (name, email) from CSV file and inserts into SQLite database.
Includes data validation, error handling, and duplicate detection.
"""

import csv
import sqlite3
import os
from typing import List, Dict, Tuple
from datetime import datetime
import re

class CSVToDatabase:
    """Class to handle CSV data import to SQLite database."""
    
    def __init__(self, db_name: str = "users.db"):
        """Initialize the database connection."""
        self.db_name = db_name
        self.stats = {
            'total_rows': 0,
            'imported_rows': 0,
            'skipped_rows': 0,
            'errors': []
        }
        self.setup_database()
    
    def setup_database(self) -> None:
        """Create the users table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    phone TEXT,
                    department TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create an index on email for faster lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_email ON users(email)')
            
            conn.commit()
            print(f"✓ Database '{self.db_name}' initialized successfully.")
        except sqlite3.Error as e:
            print(f"✗ Database error: {e}")
        finally:
            conn.close()
    
    def validate_email(self, email: str) -> bool:
        """
        Validate email format using regex.
        
        Args:
            email: Email address to validate
        
        Returns:
            True if email is valid, False otherwise
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_name(self, name: str) -> bool:
        """
        Validate name format.
        
        Args:
            name: Name to validate
        
        Returns:
            True if name is valid, False otherwise
        """
        # Name should be at least 2 characters and contain only letters, spaces, hyphens
        if not name or len(name.strip()) < 2:
            return False
        return all(c.isalpha() or c in ' -' for c in name)
    
    def clean_data(self, row: Dict) -> Tuple[Dict, List[str]]:
        """
        Clean and validate CSV row data.
        
        Args:
            row: Dictionary representing a CSV row
        
        Returns:
            Tuple of (cleaned_data, list_of_errors)
        """
        errors = []
        cleaned = {}
        
        # Extract and clean name (case-insensitive lookup)
        name = None
        for key in row.keys():
            if key.lower() == 'name':
                name = row[key].strip() if row[key] else ''
                break
        
        if not name:
            errors.append("Name is required")
        elif not self.validate_name(name):
            errors.append(f"Invalid name format: '{name}'")
        else:
            cleaned['name'] = name
        
        # Extract and clean email (case-insensitive lookup)
        email = None
        for key in row.keys():
            if key.lower() == 'email':
                email = row[key].strip().lower() if row[key] else ''
                break
        
        if not email:
            errors.append("Email is required")
        elif not self.validate_email(email):
            errors.append(f"Invalid email format: '{email}'")
        else:
            cleaned['email'] = email
        
        # Optional: Phone number (case-insensitive lookup)
        phone = None
        for key in row.keys():
            if key.lower() == 'phone':
                phone = row[key].strip() if row[key] else None
                break
        cleaned['phone'] = phone if phone else None
        
        # Optional: Department (case-insensitive lookup)
        department = None
        for key in row.keys():
            if key.lower() == 'department':
                department = row[key].strip() if row[key] else None
                break
        cleaned['department'] = department if department else None
        
        return cleaned, errors
    
    def read_csv_file(self, csv_file: str) -> List[Dict]:
        """
        Read CSV file and return list of dictionaries.
        
        Args:
            csv_file: Path to CSV file
        
        Returns:
            List of dictionaries representing CSV rows
        """
        if not os.path.exists(csv_file):
            print(f"✗ CSV file '{csv_file}' not found.")
            return []
        
        try:
            rows = []
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                # Detect delimiter (comma or semicolon)
                sample = f.read(1024)
                f.seek(0)
                delimiter = ',' if sample.count(',') > sample.count(';') else ';'
                
                reader = csv.DictReader(f, delimiter=delimiter)
                
                if reader.fieldnames is None:
                    print("✗ CSV file is empty or invalid.")
                    return []
                
                for row_num, row in enumerate(reader, start=2):  # Start from 2 (skip header)
                    rows.append(row)
                    self.stats['total_rows'] += 1
            
            print(f"✓ Successfully read {len(rows)} rows from '{csv_file}'")
            return rows
            
        except Exception as e:
            print(f"✗ Error reading CSV file: {e}")
            return []
    
    def check_user_exists(self, email: str) -> bool:
        """
        Check if a user with the given email already exists in the database.
        
        Args:
            email: Email to check
        
        Returns:
            True if user exists, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM users WHERE email = ? LIMIT 1', (email,))
            exists = cursor.fetchone() is not None
            return exists
        except sqlite3.Error:
            return False
        finally:
            conn.close()
    
    def import_from_csv(self, csv_file: str, skip_duplicates: bool = True) -> int:
        """
        Import user data from CSV file into the database.
        
        Args:
            csv_file: Path to CSV file
            skip_duplicates: If True, skip rows with duplicate emails
        
        Returns:
            Number of successfully imported rows
        """
        print(f"\nImporting data from '{csv_file}'...")
        print("-" * 60)
        
        # Read CSV file
        rows = self.read_csv_file(csv_file)
        if not rows:
            return 0
        
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            for row_num, row in enumerate(rows, start=2):  # Row numbers start from 2
                # Clean and validate data
                cleaned_data, errors = self.clean_data(row)
                
                if errors:
                    self.stats['skipped_rows'] += 1
                    error_msg = f"Row {row_num}: {'; '.join(errors)}"
                    self.stats['errors'].append(error_msg)
                    continue
                
                # Check for duplicates
                if skip_duplicates and self.check_user_exists(cleaned_data['email']):
                    self.stats['skipped_rows'] += 1
                    error_msg = f"Row {row_num}: Duplicate email '{cleaned_data['email']}'"
                    self.stats['errors'].append(error_msg)
                    continue
                
                # Insert into database
                try:
                    cursor.execute('''
                        INSERT INTO users (name, email, phone, department)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        cleaned_data['name'],
                        cleaned_data['email'],
                        cleaned_data['phone'],
                        cleaned_data['department']
                    ))
                    self.stats['imported_rows'] += 1
                    
                except sqlite3.IntegrityError:
                    self.stats['skipped_rows'] += 1
                    self.stats['errors'].append(f"Row {row_num}: Integrity constraint violation")
            
            conn.commit()
            print(f"✓ Successfully imported {self.stats['imported_rows']} users to the database.")
            return self.stats['imported_rows']
            
        except sqlite3.Error as e:
            print(f"✗ Database error: {e}")
            return 0
        finally:
            conn.close()
    
    def display_imported_users(self, limit: int = 10) -> None:
        """Display imported users from the database."""
        try:
            conn = sqlite3.connect(self.db_name)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, email, phone, department, created_date
                FROM users
                ORDER BY id DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            
            if not rows:
                print("\n✗ No users found in the database.")
                return
            
            # Display header
            print("\n" + "="*110)
            print(f"{'ID':<5} {'Name':<25} {'Email':<35} {'Phone':<15} {'Department':<15} {'Created':<15}")
            print("="*110)
            
            # Display rows
            for row in rows:
                id_val = str(row['id'])
                name = row['name'][:24] if len(row['name']) > 24 else row['name']
                email = row['email'][:34] if len(row['email']) > 34 else row['email']
                phone = row['phone'][:14] if row['phone'] else 'N/A'
                dept = row['department'][:14] if row['department'] else 'N/A'
                created = row['created_date'][:10]
                
                print(f"{id_val:<5} {name:<25} {email:<35} {phone:<15} {dept:<15} {created:<15}")
            
            print("="*110 + "\n")
            
        except sqlite3.Error as e:
            print(f"✗ Database error: {e}")
        finally:
            conn.close()
    
    def get_database_statistics(self) -> None:
        """Display import statistics."""
        print("\n" + "="*60)
        print("IMPORT STATISTICS")
        print("="*60)
        print(f"Total rows processed: {self.stats['total_rows']}")
        print(f"Successfully imported: {self.stats['imported_rows']}")
        print(f"Skipped rows: {self.stats['skipped_rows']}")
        
        if self.stats['errors']:
            print(f"\nErrors/Warnings ({len(self.stats['errors'])}):")
            for i, error in enumerate(self.stats['errors'][:10], 1):  # Show first 10 errors
                print(f"  {i}. {error}")
            if len(self.stats['errors']) > 10:
                print(f"  ... and {len(self.stats['errors']) - 10} more errors")
        
        print("="*60 + "\n")
    
    def get_user_count(self) -> int:
        """Get total number of users in database."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM users')
            count = cursor.fetchone()[0]
            return count
        except sqlite3.Error:
            return 0
        finally:
            conn.close()


def create_sample_csv(filename: str = "sample_users.csv") -> None:
    """Create a sample CSV file for testing."""
    sample_data = [
        ["Name", "Email", "Phone", "Department"],
        ["John Smith", "john.smith@example.com", "+1-555-0101", "Engineering"],
        ["Sarah Johnson", "sarah.j@company.com", "+1-555-0102", "Marketing"],
        ["Michael Brown", "mbrown@company.com", "+1-555-0103", "Sales"],
        ["Emily Davis", "emily.davis@company.com", "+1-555-0104", "HR"],
        ["David Wilson", "david.w@company.com", "+1-555-0105", "Engineering"],
        ["Jessica Anderson", "j.anderson@example.com", "+1-555-0106", "Finance"],
        ["Robert Taylor", "robert.taylor@company.com", "+1-555-0107", "Operations"],
        ["Lisa Martinez", "l.martinez@company.com", "+1-555-0108", "Engineering"],
        ["James Thompson", "jthompson@company.com", "+1-555-0109", "Support"],
        ["Jennifer Lee", "j.lee@example.com", "+1-555-0110", "Marketing"],
    ]
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(sample_data)
        print(f"✓ Sample CSV file '{filename}' created successfully.")
    except Exception as e:
        print(f"✗ Error creating sample CSV: {e}")


def main():
    """Main function to demonstrate the CSVToDatabase functionality."""
    print("\n" + "="*60)
    print("TASK 3: CSV DATA IMPORT TO DATABASE")
    print("="*60)
    
    # Create sample CSV if it doesn't exist
    csv_filename = "sample_users.csv"
    if not os.path.exists(csv_filename):
        create_sample_csv(csv_filename)
    
    # Initialize database handler
    db_handler = CSVToDatabase("users.db")
    
    # Import data from CSV
    imported_count = db_handler.import_from_csv(csv_filename, skip_duplicates=True)
    
    # Display statistics
    db_handler.get_database_statistics()
    
    # Display imported users
    print(f"Users in database: {db_handler.get_user_count()}")
    print("\nRecently imported users:")
    db_handler.display_imported_users(limit=10)
    
    print("✓ Task 3 completed successfully!")


if __name__ == "__main__":
    main()
