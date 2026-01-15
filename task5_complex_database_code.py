"""
Task 5: Most Complex Database Code - Advanced Database Operations
Demonstrates advanced database design, optimization, and operations:
- Complex schema with relationships and constraints
- Advanced SQL queries with CTEs, window functions, subqueries
- Transaction handling and ACID properties
- Database indexing and query optimization
- Data aggregation and reporting
- Stored procedures and views
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = 'pending'
    PROCESSING = 'processing'
    SHIPPED = 'shipped'
    DELIVERED = 'delivered'
    CANCELLED = 'cancelled'


@dataclass
class Customer:
    name: str
    email: str
    phone: str
    city: str
    country: str
    registration_date: str = None
    customer_id: int = None


@dataclass
class Product:
    name: str
    category: str
    price: float
    stock_quantity: int
    supplier_id: int
    product_id: int = None


class AdvancedDatabaseOperations:
    """Advanced database operations with complex queries and transactions."""
    
    def __init__(self, db_name: str = "ecommerce.db"):
        self.db_name = db_name
        self.conn = None
        self.setup_database()
    
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_name)
        self.conn.row_factory = sqlite3.Row
        # Enable foreign keys
        self.conn.execute('PRAGMA foreign_keys = ON')
        return self.conn
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def setup_database(self):
        """Create complex database schema with relationships."""
        self.connect()
        cursor = self.conn.cursor()
        
        # Suppliers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS suppliers (
                supplier_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL,
                phone TEXT,
                country TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Customers table with constraints
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT,
                city TEXT,
                country TEXT,
                registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_spent REAL DEFAULT 0.0,
                CONSTRAINT email_valid CHECK (email LIKE '%@%.%'),
                CONSTRAINT name_not_empty CHECK (length(name) > 0)
            )
        ''')
        
        # Products table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL CHECK (price > 0),
                stock_quantity INTEGER NOT NULL CHECK (stock_quantity >= 0),
                supplier_id INTEGER NOT NULL,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
                    ON DELETE CASCADE ON UPDATE CASCADE
            )
        ''')
        
        # Orders table with status tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_amount REAL NOT NULL CHECK (total_amount >= 0),
                status TEXT DEFAULT 'pending',
                shipping_address TEXT,
                estimated_delivery TIMESTAMP,
                actual_delivery TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                    ON DELETE CASCADE ON UPDATE CASCADE,
                CONSTRAINT valid_status CHECK (
                    status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled')
                )
            )
        ''')
        
        # Order items (join table for many-to-many relationship)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS order_items (
                order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL CHECK (quantity > 0),
                unit_price REAL NOT NULL CHECK (unit_price > 0),
                discount_percentage REAL DEFAULT 0.0 CHECK (discount_percentage >= 0 AND discount_percentage <= 100),
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
                    ON DELETE CASCADE ON UPDATE CASCADE,
                FOREIGN KEY (product_id) REFERENCES products(product_id)
                    ON DELETE RESTRICT ON UPDATE CASCADE,
                UNIQUE(order_id, product_id)
            )
        ''')
        
        # Reviews table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                order_id INTEGER NOT NULL,
                rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                comment TEXT,
                review_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
                FOREIGN KEY (product_id) REFERENCES products(product_id),
                FOREIGN KEY (order_id) REFERENCES orders(order_id),
                UNIQUE(customer_id, product_id, order_id)
            )
        ''')
        
        # Create indexes for query optimization
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_customer_email ON customers(email)''')
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_order_customer ON orders(customer_id)''')
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_order_date ON orders(order_date)''')
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_product_category ON products(category)''')
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id)''')
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_review_product ON reviews(product_id)''')
        
        # Create views for complex queries
        self._create_views(cursor)
        
        self.conn.commit()
        print("✓ Database schema created successfully with indexes and views.")
    
    def _create_views(self, cursor):
        """Create database views for complex queries."""
        
        # Customer spending analysis view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS customer_spending_analysis AS
            SELECT 
                c.customer_id,
                c.name,
                c.email,
                COUNT(DISTINCT o.order_id) as total_orders,
                COALESCE(SUM(o.total_amount), 0) as total_spent,
                ROUND(AVG(o.total_amount), 2) as average_order_value,
                MAX(o.order_date) as last_order_date,
                MIN(o.order_date) as first_order_date,
                ROUND((julianday(MAX(o.order_date)) - julianday(MIN(o.order_date))), 0) as days_as_customer
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY c.customer_id, c.name, c.email
        ''')
        
        # Product performance view
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS product_performance AS
            SELECT 
                p.product_id,
                p.name,
                p.category,
                p.price,
                p.stock_quantity,
                COUNT(DISTINCT oi.order_id) as total_sales,
                SUM(oi.quantity) as units_sold,
                ROUND(SUM(oi.quantity * oi.unit_price), 2) as revenue,
                ROUND(AVG(r.rating), 2) as average_rating,
                COUNT(r.review_id) as review_count
            FROM products p
            LEFT JOIN order_items oi ON p.product_id = oi.product_id
            LEFT JOIN reviews r ON p.product_id = r.product_id
            GROUP BY p.product_id, p.name, p.category, p.price, p.stock_quantity
        ''')
    
    def insert_sample_data(self):
        """Insert sample data for testing."""
        self.connect()
        cursor = self.conn.cursor()
        
        try:
            # Insert suppliers
            suppliers = [
                ('Tech Supplies Co', 'contact@techsupply.com', '+1-555-0001', 'USA'),
                ('Global Electronics', 'info@globalelec.com', '+1-555-0002', 'China'),
                ('Premium Goods Inc', 'sales@premiumgoods.com', '+1-555-0003', 'Japan'),
            ]
            
            cursor.executemany('''
                INSERT INTO suppliers (name, email, phone, country)
                VALUES (?, ?, ?, ?)
            ''', suppliers)
            
            # Insert products
            products = [
                ('Laptop Pro', 'Electronics', 1299.99, 50, 1),
                ('Wireless Mouse', 'Electronics', 29.99, 200, 2),
                ('USB Cable', 'Accessories', 12.99, 500, 1),
                ('Monitor 27inch', 'Electronics', 399.99, 30, 2),
                ('Keyboard Mechanical', 'Accessories', 149.99, 80, 3),
            ]
            
            cursor.executemany('''
                INSERT INTO products (name, category, price, stock_quantity, supplier_id)
                VALUES (?, ?, ?, ?, ?)
            ''', products)
            
            # Insert customers
            customers = [
                ('John Smith', 'john@example.com', '+1-555-1001', 'New York', 'USA'),
                ('Sarah Johnson', 'sarah@example.com', '+1-555-1002', 'Los Angeles', 'USA'),
                ('Michael Chen', 'michael@example.com', '+1-555-1003', 'San Francisco', 'USA'),
                ('Emily Davis', 'emily@example.com', '+1-555-1004', 'Chicago', 'USA'),
                ('David Wilson', 'david@example.com', '+1-555-1005', 'Houston', 'USA'),
            ]
            
            cursor.executemany('''
                INSERT INTO customers (name, email, phone, city, country)
                VALUES (?, ?, ?, ?, ?)
            ''', customers)
            
            # Insert orders with related order items
            np.random.seed(42)
            for customer_id in range(1, 6):
                num_orders = np.random.randint(1, 4)
                for _ in range(num_orders):
                    order_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
                    cursor.execute('''
                        INSERT INTO orders (customer_id, order_date, total_amount, status)
                        VALUES (?, ?, ?, ?)
                    ''', (customer_id, order_date, 0.0, 'shipped'))
                    
                    order_id = cursor.lastrowid
                    num_items = np.random.randint(1, 4)
                    total_amount = 0
                    
                    for product_id in np.random.choice(range(1, 6), num_items, replace=False):
                        cursor.execute('SELECT price FROM products WHERE product_id = ?', (product_id,))
                        price = cursor.fetchone()[0]
                        quantity = np.random.randint(1, 3)
                        discount = np.random.choice([0, 5, 10])
                        
                        item_total = quantity * price * (1 - discount/100)
                        total_amount += item_total
                        
                        cursor.execute('''
                            INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount_percentage)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (order_id, product_id, quantity, price, discount))
                    
                    cursor.execute('''
                        UPDATE orders SET total_amount = ? WHERE order_id = ?
                    ''', (total_amount, order_id))
            
            # Insert reviews
            reviews = [
                (1, 1, 1, 5, 'Excellent laptop! Very satisfied.'),
                (1, 2, 2, 4, 'Good mouse, but a bit expensive.'),
                (2, 4, 3, 5, 'Perfect monitor for my setup.'),
                (3, 1, 4, 5, 'Amazing value for the price!'),
                (4, 3, 5, 3, 'Average product, does the job.'),
            ]
            
            cursor.executemany('''
                INSERT INTO reviews (customer_id, product_id, order_id, rating, comment)
                VALUES (?, ?, ?, ?, ?)
            ''', reviews)
            
            self.conn.commit()
            print("✓ Sample data inserted successfully.")
            
        except sqlite3.IntegrityError as e:
            print(f"⚠ Sample data already exists or constraint violation: {e}")
            self.conn.rollback()
    
    def execute_complex_queries(self):
        """Execute and display results of complex queries."""
        self.connect()
        cursor = self.conn.cursor()
        
        print("\n" + "="*80)
        print("COMPLEX DATABASE QUERIES")
        print("="*80)
        
        # Query 1: Customer spending analysis with CTEs
        print("\n1. TOP CUSTOMERS BY TOTAL SPENDING (CTE Query):")
        print("-" * 80)
        
        query1 = '''
            WITH customer_stats AS (
                SELECT 
                    c.customer_id,
                    c.name,
                    COUNT(o.order_id) as order_count,
                    SUM(o.total_amount) as total_spent,
                    AVG(o.total_amount) as avg_order_value,
                    ROW_NUMBER() OVER (ORDER BY SUM(o.total_amount) DESC) as spending_rank
                FROM customers c
                LEFT JOIN orders o ON c.customer_id = o.customer_id
                GROUP BY c.customer_id, c.name
            )
            SELECT 
                spending_rank,
                name,
                order_count,
                ROUND(total_spent, 2) as total_spent,
                ROUND(avg_order_value, 2) as avg_order_value
            FROM customer_stats
            WHERE spending_rank <= 5
            ORDER BY spending_rank
        '''
        
        cursor.execute(query1)
        results = cursor.fetchall()
        for row in results:
            print(f"  Rank {row[0]}: {row[1]} | Orders: {row[2]} | Spent: ${row[3]} | Avg: ${row[4]}")
        
        # Query 2: Product ranking with window functions
        print("\n2. PRODUCT PERFORMANCE RANKING (Window Functions):")
        print("-" * 80)
        
        query2 = '''
            SELECT 
                p.product_id,
                p.name,
                SUM(oi.quantity) as units_sold,
                ROUND(SUM(oi.quantity * oi.unit_price), 2) as revenue,
                RANK() OVER (ORDER BY SUM(oi.quantity * oi.unit_price) DESC) as revenue_rank,
                ROUND(100.0 * SUM(oi.quantity * oi.unit_price) / 
                    SUM(SUM(oi.quantity * oi.unit_price)) OVER (), 2) as revenue_percentage
            FROM products p
            LEFT JOIN order_items oi ON p.product_id = oi.product_id
            GROUP BY p.product_id, p.name
            ORDER BY revenue_rank
        '''
        
        cursor.execute(query2)
        results = cursor.fetchall()
        print(f"{'Rank':<5} {'Product':<25} {'Units':<8} {'Revenue':<12} {'% of Total':<10}")
        for row in results:
            print(f"{row[4]:<5} {row[1]:<25} {row[2]:<8} ${row[3]:<11} {row[5]}%")
        
        # Query 3: Monthly sales trend with subqueries
        print("\n3. MONTHLY SALES TREND (Subquery):")
        print("-" * 80)
        
        query3 = '''
            SELECT 
                strftime('%Y-%m', order_date) as month,
                COUNT(*) as num_orders,
                ROUND(SUM(total_amount), 2) as monthly_revenue,
                ROUND(AVG(total_amount), 2) as avg_order_value
            FROM orders
            WHERE order_date IS NOT NULL
            GROUP BY strftime('%Y-%m', order_date)
            ORDER BY month DESC
        '''
        
        cursor.execute(query3)
        results = cursor.fetchall()
        print(f"{'Month':<15} {'Orders':<10} {'Revenue':<15} {'Avg Order':<15}")
        for row in results:
            print(f"{row[0]:<15} {row[1]:<10} ${row[2]:<14} ${row[3]:<14}")
        
        # Query 4: Customer segmentation
        print("\n4. CUSTOMER SEGMENTATION ANALYSIS:")
        print("-" * 80)
        
        query4 = '''
            SELECT 
                CASE 
                    WHEN SUM(o.total_amount) >= 1000 THEN 'Premium'
                    WHEN SUM(o.total_amount) >= 500 THEN 'Gold'
                    WHEN SUM(o.total_amount) >= 100 THEN 'Silver'
                    ELSE 'Bronze'
                END as customer_segment,
                COUNT(DISTINCT c.customer_id) as num_customers,
                ROUND(AVG(SUM(o.total_amount)), 2) as avg_spending,
                ROUND(SUM(SUM(o.total_amount)), 2) as total_segment_spending
            FROM customers c
            LEFT JOIN orders o ON c.customer_id = o.customer_id
            GROUP BY customer_segment
            ORDER BY total_segment_spending DESC
        '''
        
        cursor.execute(query4)
        results = cursor.fetchall()
        print(f"{'Segment':<15} {'Customers':<12} {'Avg Spending':<15} {'Total Spending':<15}")
        for row in results:
            print(f"{row[0]:<15} {row[1]:<12} ${row[2]:<14} ${row[3]:<14}")
        
        # Query 5: Orders with item details (JOIN with aggregation)
        print("\n5. ORDERS WITH ITEMIZED DETAILS (Complex JOIN):")
        print("-" * 80)
        
        query5 = '''
            SELECT 
                o.order_id,
                c.name,
                o.order_date,
                GROUP_CONCAT(p.name || ' x' || oi.quantity, ', ') as items,
                ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount_percentage/100)), 2) as order_total
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            LEFT JOIN order_items oi ON o.order_id = oi.order_id
            LEFT JOIN products p ON oi.product_id = p.product_id
            GROUP BY o.order_id, c.name, o.order_date
            ORDER BY o.order_date DESC
            LIMIT 5
        '''
        
        cursor.execute(query5)
        results = cursor.fetchall()
        for row in results:
            print(f"\n  Order {row[0]}: {row[1]}")
            print(f"    Date: {row[2]}")
            print(f"    Items: {row[3]}")
            print(f"    Total: ${row[4]}")
    
    def transaction_example(self):
        """Demonstrate ACID transactions."""
        self.connect()
        cursor = self.conn.cursor()
        
        print("\n" + "="*80)
        print("TRANSACTION EXAMPLE (ACID Properties)")
        print("="*80)
        
        try:
            # Start transaction
            cursor.execute('BEGIN TRANSACTION')
            
            # Simulate transferring products between suppliers
            cursor.execute('''
                UPDATE products SET supplier_id = 2 WHERE product_id = 1
            ''')
            
            cursor.execute('''
                UPDATE suppliers SET name = name || '_updated' WHERE supplier_id = 1
            ''')
            
            # Commit transaction
            self.conn.commit()
            print("✓ Transaction committed successfully (ACID properties maintained)")
            
        except Exception as e:
            self.conn.rollback()
            print(f"✗ Transaction rolled back: {e}")


def main():
    """Main function demonstrating complex database operations."""
    print("\n" + "="*60)
    print("TASK 5: MOST COMPLEX DATABASE CODE")
    print("Advanced Database Operations & Query Optimization")
    print("="*60)
    
    # Initialize database
    db_ops = AdvancedDatabaseOperations("ecommerce.db")
    
    # Insert sample data
    db_ops.insert_sample_data()
    
    # Execute complex queries
    db_ops.execute_complex_queries()
    
    # Demonstrate transactions
    db_ops.transaction_example()
    
    db_ops.close()
    
    print("\n✓ Advanced database operations completed successfully!")


if __name__ == "__main__":
    main()
