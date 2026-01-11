# AccuKnox AI/ML Assignment - Web UI

Complete implementation of Tasks 1, 2, and 3 with an interactive web interface for testing and demonstration.

---

## 📋 Project Structure

### **Core Task Files:**

#### `task1_api_data_retrieval.py`

- **Purpose:** Fetch books from Google Books API and store in SQLite database
- **Key Functions:**
  - `fetch_books_from_api()` - Retrieves books using Google Books API
  - `store_books_in_database()` - Saves fetched books to `books.db`
  - Validates and handles NULL values for missing fields
- **Database:** Creates/uses `books.db` with columns: id, title, author, isbn, publication_year, publisher, rating

#### `task2_data_processing_visualization.py`

- **Purpose:** Generate student score data and create visualizations
- **Key Classes:**
  - `StudentScoreAnalyzer` - Generates random student scores and creates analysis charts
- **Visualizations Generated:**
  - Score distribution dashboard (histogram, box plot, KDE plot)
  - Subject & class comparison (heatmaps and comparisons)
- **Output:** PNG images converted to base64 for web display

#### `task3_csv_import_database.py`

- **Purpose:** Import user data from CSV files with validation and duplicate detection
- **Key Functions:**
  - `validate_email()` - Email format validation using regex
  - `validate_name()` - Name format validation
  - `clean_data()` - Case-insensitive column matching and data cleaning
  - `import_from_csv()` - Processes CSV and imports to `users.db`
- **Validation:**
  - Name: Required, 2+ characters, letters/spaces/hyphens only
  - Email: Required, valid email format
  - Phone: Optional
  - Department: Optional
- **Duplicate Detection:** Skips rows with emails already in database

### **Backend:**

#### `app.py` (490 lines)

- **Framework:** Flask web application
- **Port:** http://localhost:5000
- **Routes:**

  **Task 1 (Books API):**

  - `POST /api/task1/fetch` - Fetch books from Google Books API
  - `POST /api/task1/store` - Store fetched books in database
  - `GET /api/task1/display` - Retrieve and display stored books

  **Task 2 (Data Processing):**

  - `POST /api/task2/fetch` - Generate fresh student score data
  - `POST /api/task2/visualize` - Create and return visualization charts
  - `POST /api/task2/stats` - Return statistical analysis

  **Task 3 (CSV Import):**

  - `POST /api/task3/upload` - Upload and process CSV file
  - `GET /api/task3/data` - Retrieve imported users from database
  - `GET /api/task3/download-sample` - Download sample CSV template

### **Frontend:**

#### `templates/index.html`

- **Technology:** HTML5, CSS3, Vanilla JavaScript
- **Features:**
  - Responsive design (mobile, tablet, desktop)
  - Real-time form validation
  - Modal lightbox for image viewing
  - Loading animations and error handling
  - Data statistics displayed in cards

### **Databases:**

- **`books.db`** - SQLite database storing books from Google Books API
  - Table: `books` (id, title, author, isbn, publication_year, publisher, rating)
- **`users.db`** - SQLite database storing imported users from CSV
  - Table: `users` (id, name, email, phone, department, created_date, last_updated)

---

## How to Run

### **1. Install Requirements**

```bash
pip install flask pandas matplotlib requests
```

### **2. Start the Web Application**

```bash
python app.py
```

### **3. Open in Browser**

Navigate to: **http://localhost:5000**

---

## Frontend Features & Workflow

### **Task 1: API Data Retrieval & Storage**

**Step 1: Fetch Books**

- Enter search query (e.g., "Python")
- Enter max results (e.g., 10)
- Click "Fetch Books from API"
- Results display in a table with: Title, Author, Year, Publisher
- Stats show: Total books fetched

**Step 2: Store in Database**

- Click " Store in Database"
- Books are saved to `books.db`
- Stats show: Number of books stored

**Step 3: Display from Database**

- Click " Display Results"
- Shows all books from database in table format
- Stats show: Total books, Average rating

**Frontend UI:**

- Input fields for search query and max results
- Results table with horizontal scroll (responsive)
- Stats cards displaying metrics
- Success/error alerts for user feedback

---

### **Task 2: Data Processing & Visualization**

**Step 1: Fetch Student Scores**

- Click " Fetch Student Scores"
- Generates random student score data
- **Note:** Data changes with each fetch (different random values)

**Step 2: Generate Charts**

- Click " Generate Charts"
- Creates 2 visualizations:
  - **Score Distribution Dashboard** - Histograms, box plots, KDE curves
  - **Subject & Class Comparison** - Heatmaps and comparisons
- Click on chart to enlarge in modal lightbox
- All charts display as embedded images (base64 encoded)

**Step 3: Show Statistics**

- Click " Show Statistics"
- Displays:
  - Average score across all students
  - Score by subject
  - Score by class
  - Min/Max scores
  - Standard deviation

**Frontend UI:**

- Three action buttons for fetch/visualize/stats
- Image cards with hover effects
- Modal lightbox for enlarged image viewing
- Loading spinner during processing
- Data note: "Data will be different each time you fetch"

---

### **Task 3: CSV Import with Validation**

**Step 1: Upload CSV**

- Click file upload box or drag & drop CSV
- Supported columns: Name, Email, Phone, Department
- File must be `.csv` format

**Step 2: Process & Validate**

- System validates each row:
  - Name required (2+ chars, letters only)
  - Email required (valid format)
  - Detects duplicate emails (skips if already in DB)
- Shows stats:
  - Total rows processed
  - Successfully imported
  - Rows skipped
  - Errors listed

**Step 3: View Imported Users**

- Click " Show Imported Users"
- Displays all imported users in ascending ID order (1, 2, 3...)
- Table shows: ID, Name, Email, Phone, Department

**Step 4: Download Sample**

- Click "↓ Download Sample CSV"
- Downloads template with correct format:
  ```
  Name,Email,Phone,Department
  John Smith,john.smith@example.com,+1-555-0101,Engineering
  ```

**Frontend UI:**

- Drag & drop file upload zone
- Stats cards (Total rows, Imported, Skipped)
- Error list with row numbers and messages
- Results table with responsive columns
- Download link for sample CSV

---

## Data Flow

```
┌─────────────────────────────────────────────────────┐
│          BROWSER / FRONTEND (HTML/CSS/JS)           │
└──────────────────┬──────────────────────────────────┘
                   │
                   ├─ Task 1 Routes
                   │  ├─ POST /api/task1/fetch
                   │  ├─ POST /api/task1/store
                   │  └─ GET /api/task1/display
                   │
                   ├─ Task 2 Routes
                   │  ├─ POST /api/task2/fetch
                   │  ├─ POST /api/task2/visualize
                   │  └─ POST /api/task2/stats
                   │
                   └─ Task 3 Routes
                      ├─ POST /api/task3/upload
                      ├─ GET /api/task3/data
                      └─ GET /api/task3/download-sample
                              │
┌──────────────────────────────┴────────────────────────────┐
│                    FLASK BACKEND (app.py)                 │
└──────────────────────────────┬────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
   Task 1      Task 2      Task 3
   Module      Module      Module
        │          │          │
        ├─→ Google Books API
        ├─→ StudentScoreAnalyzer (random data)
        └─→ CSV file processing
                   │
        ┌──────────┼──────────┐
        │          │          │
    books.db   (PNG images)  users.db
        │       (base64)      │
        └──────────┬──────────┘
                   │
         Returns JSON Response
                   │
         Display in Browser
```

---

## 🔍 Example API Responses

### Task 1 - Fetch Books

```json
{
  "status": "success",
  "message": "✓ 10 books found",
  "data": [
    {
      "title": "Python Programming",
      "author": "John Doe",
      "publication_year": 2023,
      "publisher": "Tech Press",
      "isbn": "978-1234567890"
    }
  ]
}
```

### Task 2 - Visualize

```json
{
  "status": "success",
  "message": "✓ Visualizations generated",
  "files": {
    "dashboard": "data:image/png;base64,iVBORw0KGgo...",
    "comparison": "data:image/png;base64,iVBORw0KGgo..."
  }
}
```

### Task 3 - Upload CSV

```json
{
  "status": "success",
  "message": "Successfully imported 5 users",
  "statistics": {
    "total_rows": 5,
    "imported_rows": 5,
    "skipped_rows": 0,
    "errors": []
  }
}
```

---

## Key Features

**Task 1:**

- Real-time API integration with Google Books
- Database persistence
- Case-insensitive field matching
- Handles NULL values gracefully

**Task 2:**

- Fresh random data on each fetch
- Professional matplotlib visualizations
- Base64 image encoding for web display
- Statistical analysis with pandas

**Task 3:**

- Email validation with regex
- Case-insensitive column matching
- Duplicate detection
- Comprehensive error reporting

**Frontend:**

- Fully responsive design
- Real-time validation
- Modal lightbox for images
- Loading animations
- Error handling with user-friendly messages
- Ascending/descending sort options
- Professional UI with gradient header

---

## Technologies Used

- **Backend:** Python, Flask
- **Databases:** SQLite
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **API:** Google Books API
- **Image Encoding:** Base64

---

## Notes

- All databases persist data between sessions
- Images are generated fresh on each visualization request
- Student scores are random and different each time
- CSV validation is case-insensitive for column names
- Tables are responsive and work on mobile devices
- Error messages provide detailed feedback for troubleshooting

---
