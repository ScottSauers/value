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
            
            tickers_no_data = []
            for ticker, ticker_stats in self.data['ticker_summary'].items():
                if concept in self.data['missing_data'].get(ticker, []):
                    tickers_no_data.append(ticker)
            
            results[concept] = {
                'total_companies': total_tickers,
                'companies_with_data_pct': has_data_pct,
                'companies_fully_missing_pct': missing_pct,
                'companies_partial_na_pct': partial_na_pct,
                'tickers_no_data': sorted(tickers_no_data),
                'tickers_no_data_count': len(tickers_no_data)
            }
        
        return results

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
        self.console.print(f"Coverage Range: {coverage_stats['min']:.1f} to {coverage_stats['max']:.1f} years")
        
        # Concept Coverage Analysis
        concept_analysis = self.analyze_concept_coverage()
        
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
        
        # Company Data Quality
        company_quality = []
        for ticker, stats in self.data['ticker_summary'].items():
            company_quality.append({
                'ticker': ticker,
                'avg_na_pct': stats.get('average_na_percentage', 0),
                'years_coverage': stats.get('years_coverage', 0)
            })
        
        company_quality.sort(key=lambda x: (x['avg_na_pct'], -x['years_coverage']))
        
        # Best and worst companies table
        quality_table = Table(title="Data Quality by Company")
        quality_table.add_column("Ticker")
        quality_table.add_column("Missing Data %", justify="right")
        quality_table.add_column("Years Coverage", justify="right")
        
        self.console.print("\n[bold]Companies with Most Complete Data[/bold]")
        for company in company_quality[:15]:
            quality_table.add_row(
                company['ticker'],
                f"{company['avg_na_pct']:.1f}%",
                f"{company['years_coverage']:.1f}"
            )
        
        self.console.print(quality_table)
        
        # Poor quality companies
        poor_quality_table = Table(title="Companies with Least Complete Data")
        poor_quality_table.add_column("Ticker")
        poor_quality_table.add_column("Missing Data %", justify="right")
        poor_quality_table.add_column("Years Coverage", justify="right")
        
        for company in company_quality[-15:]:
            poor_quality_table.add_row(
                company['ticker'],
                f"{company['avg_na_pct']:.1f}%",
                f"{company['years_coverage']:.1f}"
            )
        
        self.console.print("\n")
        self.console.print(poor_quality_table)

        # Generate missing data plots
        self.plot_na_distributions()

def main():
    analyzer = SECDataQuality('data/fundamentals/analysis/analysis_report.json')
    analyzer.print_analysis()

if __name__ == "__main__":
    main()
