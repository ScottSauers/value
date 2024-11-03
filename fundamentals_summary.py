import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from typing import Dict, List, Tuple
import warnings
import sqlite3

warnings.filterwarnings('ignore')

class SECDataQuality:
    def __init__(self, report_path: str, output_dir: str = None):
        with open(report_path, 'r') as f:
            self.data = json.load(f)
        self.output_dir = Path(output_dir) if output_dir else Path(report_path).parent
        self.console = Console()
        
        # Configure plot style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.axisbelow'] = True

    def analyze_concept_coverage(self) -> Dict:
        """Get detailed coverage statistics for each SEC concept."""
        results = {}
        
        for concept, stats in self.data['concept_summary'].items():
            if not isinstance(stats, dict):
                continue
                
            total_tickers = stats.get('total_tickers', 0)
            if total_tickers == 0:
                continue
                
            missing_pct = stats.get('missing_percentage', 0)
            partial_na_pct = stats.get('partial_na_percentage', 0)
            has_data_pct = 100 - missing_pct
            
            # Get actual data availability from granular cache
            tickers_no_data = []
            total_records = 0
            with sqlite3.connect('data/fundamentals/cache/granular_cache.db') as conn:
                # Count total records
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM concept_cache 
                    WHERE concept_tag = ? 
                    AND concept_value IS NOT NULL
                """, (concept,))
                total_records = cursor.fetchone()[0]
                
                # Get tickers with no data
                cursor = conn.execute("""
                    SELECT DISTINCT ticker FROM concept_cache 
                    WHERE concept_tag = ? 
                    AND (concept_value IS NULL OR concept_value = 'N/A')
                """, (concept,))
                tickers_no_data = [row[0] for row in cursor.fetchall()]
            
            results[concept] = {
                'total_companies': total_tickers,
                'companies_with_data_pct': has_data_pct,
                'companies_fully_missing_pct': missing_pct,
                'companies_partial_na_pct': partial_na_pct,
                'tickers_no_data': sorted(tickers_no_data),
                'tickers_no_data_count': len(tickers_no_data),
                'total_records': total_records
            }
        
        return results

    def print_concept_details(self):
        """Print detailed information about each concept's data availability."""
        with sqlite3.connect('data/fundamentals/cache/granular_cache.db') as conn:
            # Get all concepts and their counts
            cursor = conn.execute("""
                SELECT concept_tag, 
                       COUNT(DISTINCT ticker) as company_count,
                       COUNT(*) as record_count,
                       COUNT(DISTINCT filing_date) as date_count,
                       AVG(CASE WHEN concept_value IS NOT NULL THEN 1 ELSE 0 END) * 100 as data_pct
                FROM concept_cache 
                GROUP BY concept_tag
                ORDER BY record_count DESC
            """)
            
            table = Table(title="Detailed Concept Statistics")
            table.add_column("Concept")
            table.add_column("Companies", justify="right")
            table.add_column("Total Records", justify="right")
            table.add_column("Date Points", justify="right")
            table.add_column("Data %", justify="right")
            
            for row in cursor:
                table.add_row(
                    row[0],
                    str(row[1]),
                    str(row[2]),
                    str(row[3]),
                    f"{row[4]:.1f}%"
                )
            
            self.console.print(table)

    def verify_data_consistency(self):
        """Verify data consistency across different storage formats, focusing only on numerical values."""
        with sqlite3.connect('data/fundamentals/cache/granular_cache.db') as conn:
            for concept in self.data['concept_summary'].keys():
                if concept in ['ticker', 'filing_date', 'units', 'taxonomy']:
                    continue
    
                # Count unique TSV files by grouping those with the same first 6 characters
                unique_tsv_files = set()
                for file in Path('data/fundamentals').glob('*_sec_data_*.tsv'):
                    unique_tsv_files.add(file.name[:6])
                unique_tsv_count = len(unique_tsv_files)
    
                # Fetch all concept values for this concept and filter numerics
                cursor = conn.execute("""
                    SELECT DISTINCT ticker, concept_value 
                    FROM concept_cache 
                    WHERE concept_tag = ?
                """, (concept,))
                results = cursor.fetchall()
    
                # Filter only numerical values (handling commas) 
                numerical_cache_count = sum(
                    1 for _, value in results if is_numeric(value)
                )
    
                # Check for discrepancy with a threshold of 10%
                if abs(unique_tsv_count - numerical_cache_count) > unique_tsv_count * 0.1:
                    self.console.print(f"[yellow]Warning: Numerical data inconsistency for {concept}")
                    self.console.print(f"Unique TSV files: {unique_tsv_count}, Numerical cache records: {numerical_cache_count}")

        def is_numeric(value: str) -> bool:
            """Check if a string value is numeric, allowing for commas."""
            try:
                float(value.replace(",", ""))
                return True
            except ValueError:
                return False


    def plot_na_distributions(self):
        """Create enhanced visualizations of missing data distributions."""
        # Distribution of N/A percentages by ticker
        plt.figure(figsize=(12, 6))
        ticker_na_pcts = [stats.get('average_na_percentage', 0) 
                         for stats in self.data['ticker_summary'].values()]
        plt.hist(ticker_na_pcts, bins=50, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Missing Data Percentages Across Companies')
        plt.xlabel('Average Missing Data Percentage')
        plt.ylabel('Number of Companies')
        plt.axvline(np.mean(ticker_na_pcts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(ticker_na_pcts):.1f}%')
        plt.legend()
        plt.savefig(self.output_dir / 'missing_data_by_company.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Distribution of N/A percentages by concept
        plt.figure(figsize=(12, 6))
        concept_na_pcts = []
        concept_labels = []
        for concept, stats in self.data['concept_summary'].items():
            if isinstance(stats, dict):
                pct = stats.get('missing_percentage', 0) + stats.get('partial_na_percentage', 0)
                concept_na_pcts.append(pct)
                concept_labels.append(concept)
        
        plt.hist(concept_na_pcts, bins=50, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Missing Data Percentages Across Concepts')
        plt.xlabel('Total Missing Data Percentage')
        plt.ylabel('Number of Concepts')
        plt.axvline(np.mean(concept_na_pcts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(concept_na_pcts):.1f}%')
        plt.legend()
        plt.savefig(self.output_dir / 'missing_data_by_concept.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Years coverage distribution
        plt.figure(figsize=(12, 6))
        years_coverage = [stats.get('years_coverage', 0) 
                         for stats in self.data['ticker_summary'].values()]
        plt.hist(years_coverage, bins=50, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Years Coverage')
        plt.xlabel('Years of Data Available')
        plt.ylabel('Number of Companies')
        plt.axvline(np.median(years_coverage), color='red', linestyle='--', 
                   label=f'Median: {np.median(years_coverage):.1f} years')
        plt.legend()
        plt.savefig(self.output_dir / 'years_coverage.png', bbox_inches='tight', dpi=300)
        plt.close()

    def print_analysis(self):
        """Print comprehensive data quality analysis."""
        self.console.print("\n[bold]SEC Data Quality Analysis[/bold]\n")
        
        # Coverage Statistics
        coverage_stats = self.data['years_coverage_summary']
        total_companies = len(self.data['ticker_summary'])
        zero_coverage = sum(1 for stats in self.data['ticker_summary'].values() 
                          if stats.get('years_coverage', 0) == 0)
        
        self.console.print("[bold]Coverage Overview[/bold]")
        self.console.print(f"Total Companies: {total_companies}")
        self.console.print(f"Companies with Zero Coverage: {zero_coverage} ({zero_coverage/total_companies*100:.1f}%)")
        self.console.print(f"Mean Coverage: {coverage_stats['mean']:.2f} years")
        self.console.print(f"Median Coverage: {coverage_stats['median']:.2f} years")
        self.console.print(f"Coverage Range: {coverage_stats['min']:.1f} to {coverage_stats['max']:.1f} years\n")

        # Additional Coverage Statistics
        coverage_years = [stats.get('years_coverage', 0) for stats in self.data['ticker_summary'].values()]
        self.console.print("[bold]Detailed Coverage Statistics[/bold]")
        self.console.print(f"Companies with >10 years coverage: "
                         f"{sum(1 for y in coverage_years if y > 10)} "
                         f"({sum(1 for y in coverage_years if y > 10)/total_companies*100:.1f}%)")
        self.console.print(f"Companies with >5 years coverage: "
                         f"{sum(1 for y in coverage_years if y > 5)} "
                         f"({sum(1 for y in coverage_years if y > 5)/total_companies*100:.1f}%)")
        self.console.print(f"Companies with <1 year coverage: "
                         f"{sum(1 for y in coverage_years if y < 1)} "
                         f"({sum(1 for y in coverage_years if y < 1)/total_companies*100:.1f}%)\n")

        # Missing Data Statistics
        na_percentages = [stats.get('average_na_percentage', 0) for stats in self.data['ticker_summary'].values()]
        self.console.print("[bold]Missing Data Overview[/bold]")
        self.console.print(f"Average missing data across all companies: {np.mean(na_percentages):.1f}%")
        self.console.print(f"Median missing data across all companies: {np.median(na_percentages):.1f}%")
        self.console.print(f"Companies with >90% missing data: "
                         f"{sum(1 for p in na_percentages if p > 90)} "
                         f"({sum(1 for p in na_percentages if p > 90)/total_companies*100:.1f}%)")
        self.console.print(f"Companies with <50% missing data: "
                         f"{sum(1 for p in na_percentages if p < 50)} "
                         f"({sum(1 for p in na_percentages if p < 50)/total_companies*100:.1f}%)\n")

        # Data Quality Distribution
        self.console.print("[bold]Data Quality Distribution[/bold]")
        quality_ranges = [
            (0, 25, "Excellent"),
            (25, 50, "Good"),
            (50, 75, "Fair"),
            (75, 90, "Poor"),
            (90, 100, "Very Poor")
        ]
        
        for low, high, label in quality_ranges:
            count = sum(1 for p in na_percentages if low <= p < high)
            self.console.print(f"{label} ({low}-{high}% missing): {count} companies "
                             f"({count/total_companies*100:.1f}%)")
        
        # Concept Coverage Analysis
        concept_analysis = self.analyze_concept_coverage()
        
        # Calculate concept coverage statistics
        total_concepts = len([c for c in concept_analysis.keys() 
                            if c not in ['ticker', 'filing_date', 'units', 'taxonomy']])
        concepts_well_covered = sum(1 for c, stats in concept_analysis.items() 
                                  if stats['companies_with_data_pct'] > 50 
                                  and c not in ['ticker', 'filing_date', 'units', 'taxonomy'])
        
        self.console.print(f"\n[bold]Concept Coverage Overview[/bold]")
        self.console.print(f"Total concepts tracked: {total_concepts}")
        self.console.print(f"Concepts with >50% company coverage: {concepts_well_covered} "
                         f"({concepts_well_covered/total_concepts*100:.1f}%)")
        
        table = Table(title="Concept Data Availability")
        table.add_column("Concept", style="cyan")
        table.add_column("Has Any Data %", justify="right")
        table.add_column("Fully Missing %", justify="right")
        table.add_column("Partial Data %", justify="right")
        table.add_column("Total Companies", justify="right")
        
        # Sort concepts by data availability
        sorted_concepts = sorted(concept_analysis.items(),
                               key=lambda x: x[1]['companies_with_data_pct'],
                               reverse=True)
        
        for concept, stats in sorted_concepts:
            if concept in ['ticker', 'filing_date', 'units', 'taxonomy']:
                continue
                
            table.add_row(
                concept,
                f"{stats['companies_with_data_pct']:.1f}%",
                f"{stats['companies_fully_missing_pct']:.1f}%",
                f"{stats['companies_partial_na_pct']:.1f}%",
                str(stats['total_companies'])
            )
        
        self.console.print("\n")
        self.console.print(table)
        
        # Company Data Quality Analysis
        company_quality = []
        for ticker, stats in self.data['ticker_summary'].items():
            company_quality.append({
                'ticker': ticker,
                'avg_na_pct': stats.get('average_na_percentage', 0),
                'years_coverage': stats.get('years_coverage', 0),
                'missing_concepts': stats.get('missing_concepts', 0)
            })
        
        company_quality.sort(key=lambda x: (x['avg_na_pct'], -x['years_coverage']))
        
        # Best and worst companies table
        quality_table = Table(title="Companies with Most Complete Data")
        quality_table.add_column("Ticker")
        quality_table.add_column("Missing Data %", justify="right")
        quality_table.add_column("Years Coverage", justify="right")
        quality_table.add_column("Missing Concepts", justify="right")
        
        self.console.print("\n[bold]High Quality Data Companies[/bold]")
        for company in company_quality[:15]:
            quality_table.add_row(
                company['ticker'],
                f"{company['avg_na_pct']:.1f}%",
                f"{company['years_coverage']:.1f}",
                str(company['missing_concepts'])
            )
        
        self.console.print(quality_table)
        
        # Poor quality companies
        poor_quality_table = Table(title="Companies with Least Complete Data")
        poor_quality_table.add_column("Ticker")
        poor_quality_table.add_column("Missing Data %", justify="right")
        poor_quality_table.add_column("Years Coverage", justify="right")
        poor_quality_table.add_column("Missing Concepts", justify="right")
        
        for company in company_quality[-15:]:
            poor_quality_table.add_row(
                company['ticker'],
                f"{company['avg_na_pct']:.1f}%",
                f"{company['years_coverage']:.1f}",
                str(company['missing_concepts'])
            )
        
        self.console.print("\n")
        self.console.print(poor_quality_table)

        # Data Age Analysis
        recent_data = {ticker: stats for ticker, stats in self.data['ticker_summary'].items()
                      if stats.get('years_coverage', 0) > 0}
        if recent_data:
            self.console.print("\n[bold]Data Recency Analysis[/bold]")
            active_companies = len(recent_data)
            self.console.print(f"Companies with recent data: {active_companies} "
                             f"({active_companies/total_companies*100:.1f}%)")

        # Generate missing data plots
        self.plot_na_distributions()

def main():
    analyzer = SECDataQuality('data/fundamentals/analysis/analysis_report.json')
    analyzer.print_analysis()
    analyzer.print_concept_details()
    analyzer.verify_data_consistency()

if __name__ == "__main__":
    main()
