"""
Task 2: Data Processing and Visualization
Fetches student test score data from API, calculates average score, 
and creates visualizations using matplotlib and seaborn.
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import json

class StudentScoreAnalyzer:
    """Class to handle student test score data processing and visualization."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.scores_data = []
        self.df = None
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
    
    def fetch_student_scores_from_api(self) -> List[Dict]:
        """
        Fetch student test score data from a public API.
        Using JSONPlaceholder API to simulate test scores.
        
        Returns:
            List of student score dictionaries
        """
        try:
            # Using JSONPlaceholder Users API and simulating test scores
            response = requests.get('https://jsonplaceholder.typicode.com/users', timeout=10)
            response.raise_for_status()
            
            users = response.json()
            scores_data = []
            
            # Generate realistic test scores based on user data
            # No fixed seed - generates different data each time
            subjects = ['Mathematics', 'Science', 'English', 'History', 'Computer Science']
            
            for user in users:
                for i, subject in enumerate(subjects):
                    # Generate scores with realistic distribution
                    base_score = np.random.normal(75, 12)
                    score = max(0, min(100, base_score + np.random.normal(0, 5)))
                    
                    score_entry = {
                        'student_id': user['id'],
                        'student_name': user['name'],
                        'subject': subject,
                        'test_score': round(score, 2),
                        'test_date': f"2024-Q{(i % 4) + 1}",
                        'class': f"Class {chr(65 + (user['id'] - 1) % 3)}",
                        'teacher': user['company']['name'][:20]
                    }
                    scores_data.append(score_entry)
            
            print(f"✓ Successfully fetched scores for {len(users)} students across {len(subjects)} subjects.")
            self.scores_data = scores_data
            return scores_data
            
        except requests.exceptions.RequestException as e:
            print(f"✗ API request error: {e}")
            return []
    
    def process_scores(self) -> pd.DataFrame:
        """
        Process the fetched scores data into a DataFrame.
        
        Returns:
            Processed pandas DataFrame
        """
        if not self.scores_data:
            print("✗ No scores data to process.")
            return None
        
        try:
            self.df = pd.DataFrame(self.scores_data)
            
            # Data validation and cleaning
            self.df['test_score'] = pd.to_numeric(self.df['test_score'], errors='coerce')
            self.df = self.df.dropna(subset=['test_score'])
            
            print(f"✓ Processed {len(self.df)} score entries successfully.")
            return self.df
            
        except Exception as e:
            print(f"✗ Error processing scores: {e}")
            return None
    
    def calculate_statistics(self) -> Dict:
        """
        Calculate statistical measures from the scores.
        
        Returns:
            Dictionary containing various statistics
        """
        if self.df is None or len(self.df) == 0:
            print("✗ No data available for statistics calculation.")
            return {}
        
        try:
            stats = {
                'overall_average': self.df['test_score'].mean(),
                'overall_median': self.df['test_score'].median(),
                'overall_std': self.df['test_score'].std(),
                'overall_min': self.df['test_score'].min(),
                'overall_max': self.df['test_score'].max(),
                'subject_average': self.df.groupby('subject')['test_score'].mean().to_dict(),
                'class_average': self.df.groupby('class')['test_score'].mean().to_dict(),
                'student_average': self.df.groupby('student_name')['test_score'].mean().to_dict()
            }
            
            print("\n" + "="*60)
            print("STATISTICS SUMMARY")
            print("="*60)
            print(f"Overall Average Score: {stats['overall_average']:.2f}")
            print(f"Median Score: {stats['overall_median']:.2f}")
            print(f"Standard Deviation: {stats['overall_std']:.2f}")
            print(f"Score Range: {stats['overall_min']:.2f} - {stats['overall_max']:.2f}")
            print("="*60 + "\n")
            
            return stats
            
        except Exception as e:
            print(f"✗ Error calculating statistics: {e}")
            return {}
    
    def create_visualizations(self, output_dir: str = ".") -> None:
        """
        Create multiple visualizations of the score data.
        
        Args:
            output_dir: Directory to save visualization files
        """
        if self.df is None or len(self.df) == 0:
            print("✗ No data available for visualization.")
            return
        
        try:
            # Use faster rendering backend FIRST
            import matplotlib
            matplotlib.use('Agg')
            
            # 1. Bar Chart: Average Score by Subject
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            # Removed suptitle - using HTML card heading instead
            
            # Add proper spacing
            plt.subplots_adjust(top=0.95, bottom=0.1, left=0.08, right=0.95, hspace=0.35, wspace=0.3)
            
            # Chart 1: Average Score by Subject
            subject_avg = self.df.groupby('subject')['test_score'].mean().sort_values(ascending=False)
            colors = plt.cm.viridis(np.linspace(0, 1, len(subject_avg)))
            axes[0, 0].bar(subject_avg.index, subject_avg.values, color=colors, edgecolor='black', linewidth=1.2)
            axes[0, 0].set_title('Average Score by Subject', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('Average Score', fontsize=10)
            axes[0, 0].set_xlabel('Subject', fontsize=10)
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(subject_avg.values):
                axes[0, 0].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Chart 2: Distribution of Test Scores (Histogram)
            axes[0, 1].hist(self.df['test_score'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            axes[0, 1].axvline(self.df['test_score'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {self.df['test_score'].mean():.2f}")
            axes[0, 1].axvline(self.df['test_score'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {self.df['test_score'].median():.2f}")
            axes[0, 1].set_title('Distribution of Test Scores', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Test Score', fontsize=10)
            axes[0, 1].set_ylabel('Frequency', fontsize=10)
            axes[0, 1].legend()
            axes[0, 1].grid(axis='y', alpha=0.3)
            
            # Chart 3: Box Plot by Class
            class_data = [self.df[self.df['class'] == c]['test_score'].values 
                         for c in sorted(self.df['class'].unique())]
            bp = axes[1, 0].boxplot(class_data, labels=sorted(self.df['class'].unique()), 
                                     patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            axes[1, 0].set_title('Score Distribution by Class', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Test Score', fontsize=10)
            axes[1, 0].set_xlabel('Class', fontsize=10)
            axes[1, 0].grid(axis='y', alpha=0.3)
            
            # Chart 4: Average Score by Class (Bar Chart)
            class_avg = self.df.groupby('class')['test_score'].mean().sort_values(ascending=False)
            bars = axes[1, 1].bar(range(len(class_avg)), class_avg.values, color=colors, edgecolor='black', linewidth=1.2)
            axes[1, 1].set_xticks(range(len(class_avg)))
            axes[1, 1].set_xticklabels(class_avg.index)
            axes[1, 1].set_title('Average Score by Class', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Average Score', fontsize=10)
            axes[1, 1].set_xlabel('Class', fontsize=10)
            axes[1, 1].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(class_avg.values):
                axes[1, 1].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save the visualization (lower DPI for faster rendering)
            output_file = f"{output_dir}/student_scores_dashboard.png"
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            print(f"✓ Dashboard saved as '{output_file}'")
            plt.close()
            
            # Additional: Create a detailed subject comparison chart
            fig, ax = plt.subplots(figsize=(14, 7))
            
            subject_class_avg = self.df.groupby(['subject', 'class'])['test_score'].mean().unstack()
            subject_class_avg.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1.2)
            
            ax.set_title('Average Score by Subject and Class', fontsize=14, fontweight='bold')
            ax.set_xlabel('Subject', fontsize=11)
            ax.set_ylabel('Average Score', fontsize=11)
            ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            output_file2 = f"{output_dir}/subject_class_comparison.png"
            plt.savefig(output_file2, dpi=100, bbox_inches='tight')
            print(f"✓ Comparison chart saved as '{output_file2}'")
            plt.close()
            
        except Exception as e:
            print(f"✗ Error creating visualizations: {e}")
    
    def export_to_csv(self, filename: str = "student_scores.csv") -> None:
        """Export processed data to CSV."""
        if self.df is None:
            print("✗ No data to export.")
            return
        
        try:
            self.df.to_csv(filename, index=False)
            print(f"✓ Data exported to '{filename}'")
        except Exception as e:
            print(f"✗ Error exporting data: {e}")


def main():
    """Main function to demonstrate the StudentScoreAnalyzer functionality."""
    print("\n" + "="*60)
    print("TASK 2: DATA PROCESSING AND VISUALIZATION")
    print("="*60 + "\n")
    
    # Initialize analyzer
    analyzer = StudentScoreAnalyzer()
    
    # Fetch data from API
    print("Fetching student test scores from API...")
    analyzer.fetch_student_scores_from_api()
    
    # Process the data
    print("\nProcessing score data...")
    analyzer.process_scores()
    
    # Calculate statistics
    print("Calculating statistics...")
    stats = analyzer.calculate_statistics()
    
    # Create visualizations
    print("Creating visualizations...")
    analyzer.create_visualizations()
    
    # Export to CSV
    analyzer.export_to_csv("student_scores.csv")
    
    print("\n✓ Task 2 completed successfully!")


if __name__ == "__main__":
    main()
