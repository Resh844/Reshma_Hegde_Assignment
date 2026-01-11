"""
Task 1: API Data Retrieval and Storage
Fetches book data from Google Books API, stores in SQLite database, and displays it.
"""

import requests
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

class BookAPIDatabase:
    """Class to handle API data retrieval and SQLite database operations for books."""
    
    def __init__(self, db_name: str = "books.db"):
        """Initialize the database connection."""
        self.db_name = db_name
        self.api_url = "https://www.googleapis.com/books/v1/volumes"
        self.setup_database()
    
    def setup_database(self) -> None:
        """Create the books table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS books (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    author TEXT,
                    publication_year INTEGER,
                    publisher TEXT,
                    description TEXT,
                    isbn TEXT,
                    page_count INTEGER,
                    language TEXT,
                    categories TEXT,
                    average_rating REAL,
                    ratings_count INTEGER,
                    fetched_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            print(f"✓ Database '{self.db_name}' initialized successfully.")
        except sqlite3.Error as e:
            print(f"✗ Database error: {e}")
        finally:
            conn.close()
    
    def fetch_books_from_api(self, query: str = "python programming", max_results: int = 10) -> List[Dict]:
        """
        Fetch book data from Google Books API.
        
        Args:
            query: Search query for books
            max_results: Maximum number of results to fetch (max 40 per API)
        
        Returns:
            List of book dictionaries
        """
        try:
            params = {
                'q': query,
                'maxResults': min(max_results, 40),
                'printType': 'books'
            }
            
            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            books = []
            
            if 'items' in data:
                for item in data['items']:
                    volume_info = item.get('volumeInfo', {})
                    
                    book = {
                        'title': volume_info.get('title', 'N/A'),
                        'author': ', '.join(volume_info.get('authors', ['N/A'])),
                        'publication_year': volume_info.get('publishedDate', 'N/A')[:4] if volume_info.get('publishedDate') else None,
                        'publisher': volume_info.get('publisher', 'N/A'),
                        'description': volume_info.get('description', 'N/A'),
                        'isbn': volume_info.get('industryIdentifiers', [{}])[0].get('identifier', 'N/A') if volume_info.get('industryIdentifiers') else 'N/A',
                        'page_count': volume_info.get('pageCount'),
                        'language': volume_info.get('language', 'N/A'),
                        'categories': ', '.join(volume_info.get('categories', [])),
                        'average_rating': volume_info.get('averageRating'),
                        'ratings_count': volume_info.get('ratingsCount')
                    }
                    
                    books.append(book)
            
            print(f"✓ Successfully fetched {len(books)} books from API.")
            return books
            
        except requests.exceptions.RequestException as e:
            print(f"✗ API request error: {e}")
            return []
    
    def store_books_in_database(self, books: List[Dict]) -> int:
        """
        Store fetched books in SQLite database.
        
        Args:
            books: List of book dictionaries to store
        
        Returns:
            Number of books successfully stored
        """
        if not books:
            print("✗ No books to store.")
            return 0
        
        conn = None
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            stored_count = 0
            for book in books:
                try:
                    # Use .get() for safe dictionary access with defaults
                    cursor.execute('''
                        INSERT INTO books (title, author, publication_year, publisher, 
                                         description, isbn, page_count, language, 
                                         categories, average_rating, ratings_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        book.get('title', 'N/A'),
                        book.get('author', 'Unknown'),
                        book.get('publication_year', None),
                        book.get('publisher', 'N/A'),
                        book.get('description', 'N/A'),
                        book.get('isbn', None),
                        book.get('page_count', None),
                        book.get('language', 'en'),
                        book.get('categories', 'N/A'),
                        book.get('average_rating', None),
                        book.get('ratings_count', None)
                    ))
                    stored_count += 1
                except sqlite3.IntegrityError as ie:
                    # Book already exists (duplicate ISBN)
                    print(f"   Skipping duplicate: {book.get('title', 'Unknown')}")
                    continue
                except Exception as e:
                    print(f"   Error storing book: {str(e)}")
                    continue
            
            conn.commit()
            print(f"✓ Successfully stored {stored_count} books in the database.")
            return stored_count
            
        except sqlite3.Error as e:
            print(f"✗ Database error while storing books: {e}")
            return 0
        except Exception as e:
            print(f"✗ Error: {e}")
            return 0
        finally:
            if conn:
                conn.close()
    
    def display_books(self, limit: int = 5) -> None:
        """Display books from the database in a formatted table."""
        try:
            conn = sqlite3.connect(self.db_name)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT title, author, publication_year, publisher, 
                       page_count, average_rating
                FROM books
                ORDER BY fetched_date DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            
            if not rows:
                print("✗ No books found in the database.")
                return
            
            # Display header
            print("\n" + "="*120)
            print(f"{'Title':<35} {'Author':<25} {'Year':<6} {'Publisher':<20} {'Pages':<7} {'Rating':<7}")
            print("="*120)
            
            # Display rows
            for row in rows:
                title = row['title'][:34] if len(row['title']) > 34 else row['title']
                author = row['author'][:24] if len(row['author']) > 24 else row['author']
                year = str(row['publication_year']) if row['publication_year'] else 'N/A'
                publisher = row['publisher'][:19] if len(row['publisher']) > 19 else row['publisher']
                pages = str(row['page_count']) if row['page_count'] else 'N/A'
                rating = f"{row['average_rating']:.1f}" if row['average_rating'] else 'N/A'
                
                print(f"{title:<35} {author:<25} {year:<6} {publisher:<20} {pages:<7} {rating:<7}")
            
            print("="*120 + "\n")
            
            # Display summary statistics
            cursor.execute('SELECT COUNT(*) as total FROM books')
            total = cursor.fetchone()['total']
            print(f"Total books in database: {total}")
            
        except sqlite3.Error as e:
            print(f"✗ Database error while displaying books: {e}")
        finally:
            conn.close()
    
    def get_database_statistics(self) -> None:
        """Display database statistics."""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) as total FROM books')
            total = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(average_rating) as avg_rating FROM books WHERE average_rating IS NOT NULL')
            avg_rating = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(page_count) as avg_pages FROM books WHERE page_count IS NOT NULL')
            avg_pages = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT language) as languages FROM books')
            languages = cursor.fetchone()[0]
            
            print("\n" + "="*50)
            print("DATABASE STATISTICS")
            print("="*50)
            print(f"Total Books: {total}")
            print(f"Average Rating: {avg_rating:.2f if avg_rating else 'N/A'}")
            print(f"Average Pages: {avg_pages:.0f if avg_pages else 'N/A'}")
            print(f"Languages: {languages}")
            print("="*50 + "\n")
            
        except sqlite3.Error as e:
            print(f"✗ Database error while fetching statistics: {e}")
        finally:
            conn.close()


def main():
    """Main function to demonstrate the BookAPIDatabase functionality."""
    print("\n" + "="*60)
    print("TASK 1: API DATA RETRIEVAL AND STORAGE")
    print("="*60 + "\n")
    
    # Initialize the database
    db = BookAPIDatabase("books.db")
    
    # Fetch books from API
    print("Fetching books from Google Books API...")
    books = db.fetch_books_from_api(query="python programming", max_results=15)
    
    # Store in database
    if books:
        db.store_books_in_database(books)
    
    # Display the retrieved data
    print("\nDisplaying retrieved books:")
    db.display_books(limit=10)
    
    # Show statistics
    db.get_database_statistics()


if __name__ == "__main__":
    main()
