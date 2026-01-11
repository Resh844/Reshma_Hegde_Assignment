"""
Web UI for AccuKnox AI/ML Trainee Assignment - Tasks 1-3
Flask-based dashboard for API integration, data visualization, and CSV import
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from datetime import datetime
import io
import csv

# Import task modules
import sys
sys.path.insert(0, os.path.dirname(__file__))

from task1_api_data_retrieval import BookAPIDatabase
from task2_data_processing_visualization import StudentScoreAnalyzer
from task3_csv_import_database import CSVToDatabase

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global instances
book_db = BookAPIDatabase("books.db")
csv_db = CSVToDatabase("users.db")
# Note: score_analyzer is created per request to get fresh data

# Store results in memory for display
results = {
    'task1': None,
    'task2': None,
    'task3': None
}


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/task1/fetch', methods=['POST'])
def task1_fetch():
    """Task 1: Fetch books from API"""
    try:
        query = request.json.get('query', 'python programming')
        max_results = request.json.get('max_results', 10)
        
        # Fetch books
        books = book_db.fetch_books_from_api(query=query, max_results=max_results)
        
        if not books:
            return jsonify({'status': 'error', 'message': 'Failed to fetch books'}), 400
        
        # Store books data
        results['task1'] = {
            'books': books,
            'count': len(books),
            'query': query,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'message': f'Successfully fetched {len(books)} books',
            'data': books
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/task1/store', methods=['POST'])
def task1_store():
    """Task 1: Store books in database"""
    try:
        if 'task1' not in results or not results['task1']:
            return jsonify({'status': 'error', 'message': 'No books fetched yet. Fetch books first.'}), 400
        
        books = results['task1'].get('books', [])
        
        if not books:
            return jsonify({'status': 'error', 'message': 'No books data available'}), 400
        
        print(f"Storing {len(books)} books from API...")
        stored_count = book_db.store_books_in_database(books)
        
        print(f"Stored {stored_count} books in database")
        
        return jsonify({
            'status': 'success',
            'message': f'Successfully stored {stored_count} books in database',
            'count': stored_count
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in task1_store: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/task1/display', methods=['GET'])
def task1_display():
    """Task 1: Get books from database"""
    try:
        import sqlite3
        conn = sqlite3.connect('books.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, author, publication_year, publisher, page_count, average_rating
            FROM books ORDER BY fetched_date DESC LIMIT 10
        ''')
        
        rows = cursor.fetchall()
        books_list = []
        
        for row in rows:
            books_list.append({
                'title': row['title'],
                'author': row['author'],
                'year': row['publication_year'],
                'publisher': row['publisher'],
                'pages': row['page_count'],
                'rating': row['average_rating']
            })
        
        # Get statistics
        cursor.execute('SELECT COUNT(*) as total FROM books')
        total = cursor.fetchone()['total']
        
        cursor.execute('SELECT AVG(average_rating) as avg_rating FROM books WHERE average_rating IS NOT NULL')
        avg_rating = cursor.fetchone()['avg_rating']
        
        conn.close()
        
        return jsonify({
            'status': 'success',
            'books': books_list,
            'statistics': {
                'total_books': total,
                'average_rating': round(avg_rating, 2) if avg_rating else 0
            }
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/task2/fetch', methods=['POST'])
def task2_fetch():
    """Task 2: Fetch and process student scores"""
    try:
        # Create a NEW instance each time to get fresh random data
        score_analyzer = StudentScoreAnalyzer()
        
        # Fetch data
        score_analyzer.fetch_student_scores_from_api()
        
        if not score_analyzer.scores_data:
            return jsonify({'status': 'error', 'message': 'Failed to fetch scores'}), 400
        
        # Process data
        score_analyzer.process_scores()
        
        # Calculate statistics
        stats = score_analyzer.calculate_statistics()
        
        # Store in global for visualization and stats endpoints
        results['task2'] = {
            'analyzer': score_analyzer,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Successfully fetched and processed scores',
            'statistics': {
                'overall_average': round(stats['overall_average'], 2),
                'overall_median': round(stats['overall_median'], 2),
                'overall_std': round(stats['overall_std'], 2),
                'min_score': round(stats['overall_min'], 2),
                'max_score': round(stats['overall_max'], 2),
                'subject_averages': {k: round(v, 2) for k, v in stats['subject_average'].items()},
                'class_averages': {k: round(v, 2) for k, v in stats['class_average'].items()}
            }
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/task2/visualize', methods=['POST'])
def task2_visualize():
    """Task 2: Generate visualizations"""
    try:
        # Get the analyzer from results (stored in task2_fetch)
        if 'task2' not in results or results['task2'] is None or 'analyzer' not in results['task2']:
            return jsonify({'status': 'error', 'message': 'No data to visualize. Fetch data first.'}), 400
        
        score_analyzer = results['task2'].get('analyzer')
        if score_analyzer is None:
            return jsonify({'status': 'error', 'message': 'No data to visualize. Fetch data first.'}), 400
        
        if score_analyzer.df is None or len(score_analyzer.df) == 0:
            return jsonify({'status': 'error', 'message': 'No data to visualize. Fetch data first.'}), 400
        
        print("Creating visualizations...")
        # Create visualizations
        score_analyzer.create_visualizations()
        score_analyzer.export_to_csv("student_scores.csv")
        
        # Read images and convert to base64
        import base64
        
        dashboard_img = None
        comparison_img = None
        
        dashboard_path = os.path.join(os.getcwd(), 'student_scores_dashboard.png')
        comparison_path = os.path.join(os.getcwd(), 'subject_class_comparison.png')
        
        print(f"Looking for dashboard at: {dashboard_path}")
        print(f"Looking for comparison at: {comparison_path}")
        print(f"Dashboard exists: {os.path.exists(dashboard_path)}")
        print(f"Comparison exists: {os.path.exists(comparison_path)}")
        
        if os.path.exists(dashboard_path):
            try:
                with open(dashboard_path, 'rb') as f:
                    dashboard_img = base64.b64encode(f.read()).decode('utf-8')
                    print(f"Dashboard encoded: {len(dashboard_img)} chars")
            except Exception as e:
                print(f"Error encoding dashboard: {e}")
        
        if os.path.exists(comparison_path):
            try:
                with open(comparison_path, 'rb') as f:
                    comparison_img = base64.b64encode(f.read()).decode('utf-8')
                    print(f"Comparison encoded: {len(comparison_img)} chars")
            except Exception as e:
                print(f"Error encoding comparison: {e}")
        
        if not dashboard_img or not comparison_img:
            return jsonify({
                'status': 'error', 
                'message': f'Images not generated properly. Dashboard: {bool(dashboard_img)}, Comparison: {bool(comparison_img)}'
            }), 400
        
        return jsonify({
            'status': 'success',
            'message': 'Visualizations created successfully',
            'files': {
                'dashboard': f"data:image/png;base64,{dashboard_img}",
                'comparison': f"data:image/png;base64,{comparison_img}",
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/task2/image/<image_type>')
def task2_image(image_type):
    """Serve visualization images"""
    try:
        if image_type == 'dashboard':
            filename = 'student_scores_dashboard.png'
        elif image_type == 'comparison':
            filename = 'subject_class_comparison.png'
        else:
            return jsonify({'status': 'error', 'message': 'Invalid image type'}), 404
        
        filepath = os.path.join(os.getcwd(), filename)
        if os.path.exists(filepath):
            response = send_file(filepath, mimetype='image/png')
            # Prevent caching
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        else:
            print(f"Image not found: {filepath}")
            return jsonify({'status': 'error', 'message': f'Image not found: {filename}'}), 404
    
    except Exception as e:
        print(f"Error serving image: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/task2/stats', methods=['GET'])
def task2_stats():
    """Task 2: Get statistics"""
    try:
        if 'task2' not in results or results['task2'] is None or 'stats' not in results['task2']:
            return jsonify({'status': 'error', 'message': 'No data available. Fetch data first.'}), 400
        
        stats = results['task2'].get('stats', {})
        
        if not stats:
            return jsonify({'status': 'error', 'message': 'No data available. Fetch data first.'}), 400
        
        # Safe dictionary access with defaults
        subject_breakdown = stats.get('subject_average', {})
        class_breakdown = stats.get('class_average', {})
        
        # Handle None values
        if subject_breakdown is None:
            subject_breakdown = {}
        if class_breakdown is None:
            class_breakdown = {}
        
        return jsonify({
            'status': 'success',
            'statistics': {
                'overall_average': round(stats.get('overall_average', 0), 2),
                'overall_median': round(stats.get('overall_median', 0), 2),
                'std_deviation': round(stats.get('overall_std', 0), 2),
                'score_range': f"{round(stats.get('overall_min', 0), 2)} - {round(stats.get('overall_max', 0), 2)}",
                'subject_breakdown': {k: round(v, 2) for k, v in subject_breakdown.items()} if subject_breakdown else {},
                'class_breakdown': {k: round(v, 2) for k, v in class_breakdown.items()} if class_breakdown else {}
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/task3/upload', methods=['POST'])
def task3_upload():
    """Task 3: Upload and import CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'status': 'error', 'message': 'File must be CSV format'}), 400
        
        # Save uploaded file
        filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file.save(filename)
        
        # Reset stats for new upload and import from CSV
        csv_db.stats = {
            'total_rows': 0,
            'imported_rows': 0,
            'skipped_rows': 0,
            'errors': []
        }
        imported_count = csv_db.import_from_csv(filename, skip_duplicates=True)
        
        results['task3'] = {
            'filename': filename,
            'imported_count': imported_count,
            'stats': csv_db.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'message': f'Successfully imported {imported_count} users',
            'statistics': {
                'total_rows': csv_db.stats['total_rows'],
                'imported_rows': csv_db.stats['imported_rows'],
                'skipped_rows': csv_db.stats['skipped_rows'],
                'errors': csv_db.stats['errors'][:5]  # First 5 errors
            }
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/task3/data', methods=['GET'])
def task3_data():
    """Task 3: Get imported user data"""
    try:
        import sqlite3
        conn = sqlite3.connect('users.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, email, phone, department, created_date
            FROM users ORDER BY id ASC LIMIT 10
        ''')
        
        rows = cursor.fetchall()
        users_list = []
        
        for row in rows:
            users_list.append({
                'id': row['id'],
                'name': row['name'],
                'email': row['email'],
                'phone': row['phone'] or 'N/A',
                'department': row['department'] or 'N/A',
                'created_date': row['created_date'][:10]
            })
        
        # Get statistics
        cursor.execute('SELECT COUNT(*) as total FROM users')
        total = cursor.fetchone()['total']
        
        conn.close()
        
        return jsonify({
            'status': 'success',
            'users': users_list,
            'statistics': {
                'total_users': total
            }
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/task3/download-sample', methods=['GET'])
def task3_download_sample():
    """Task 3: Download sample CSV"""
    try:
        sample_data = [
            ["Name", "Email", "Phone", "Department"],
            ["John Smith", "john.smith@example.com", "+1-555-0101", "Engineering"],
            ["Sarah Johnson", "sarah.j@company.com", "+1-555-0102", "Marketing"],
            ["Michael Brown", "mbrown@company.com", "+1-555-0103", "Sales"],
            ["Emily Davis", "emily.davis@company.com", "+1-555-0104", "HR"],
            ["David Wilson", "david.w@company.com", "+1-555-0105", "Engineering"],
        ]
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(sample_data)
        
        # Convert to bytes
        output.seek(0)
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        
        return send_file(
            mem,
            mimetype='text/csv',
            as_attachment=True,
            download_name='sample_users.csv'
        )
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get overall status"""
    return jsonify({
        'status': 'success',
        'tasks_completed': {
            'task1': results['task1'] is not None,
            'task2': results['task2'] is not None,
            'task3': results['task3'] is not None
        }
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("AccuKnox AI/ML Trainee - Web UI Dashboard")
    print("="*60)
    print("\n✓ Starting Flask server...")
    print("✓ Access the UI at: http://localhost:5000")
    print("✓ Tasks available: 1 (API), 2 (Visualization), 3 (CSV Import)")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='localhost', port=5000)
