import os
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class SECDataAnalyzer:
    """Analyzes SEC fundamental data quality and completeness."""
    
    def __init__(self, data_dir: str = "./data/fundamentals"):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / 'cache'
        self.output_dir = self.data_dir / 'analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.console = Console()
        
        # Initialize results storage
        self.results = {
            'missing_data': defaultdict(list),
            'na_percentages': defaultdict(dict),
            'units': defaultdict(set),
            'years_coverage': defaultdict(int),
            'concept_stats': defaultdict(lambda: {
                'missing_count': 0,
                'na_count': 0,
                'total_count': 0,
                'units': set()
            })
        }

    def load_company_data(self) -> Tuple[List[Path], List[Path]]:
        """Find all TSV and JSON files in the data directory."""
        tsv_files = list(self.data_dir.glob('**/*_sec_data_*.tsv'))
        json_files = list(self.data_dir.glob('**/*_sec_data_*.json'))
        
        self.logger.info(f"Found {len(tsv_files)} TSV files and {len(json_files)} JSON files")
        return tsv_files, json_files

    def _safe_parse_date(self, date_str: str) -> pd.Timestamp:
        """Safely parse date string with multiple fallback formats."""
        try:
            return pd.to_datetime(date_str, format='mixed')
        except:
            try:
                # Try parsing just the date part if there's a time component
                date_part = date_str.split()[0]
                return pd.to_datetime(date_part)
            except:
                self.logger.warning(f"Could not parse date: {date_str}")
                return None

    def analyze_file(self, tsv_path: Path, metadata_path: Path = None) -> Dict:
        """Analyze a single company's data file."""
        try:
            # Load data
            df = pd.read_csv(tsv_path, sep='\t')
            ticker = tsv_path.stem.split('_')[0]
            
            # Load metadata if available
            metadata = None
            if metadata_path and metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            
            # Analyze missing and N/A data
            missing_cols = df.columns[df.isnull().any()].tolist()
            na_percentages = (df.isna().sum() / len(df) * 100).to_dict()
            
            # Get date range coverage
            # Parse dates with fallback
            df['filing_date'] = df['filing_date'].apply(self._safe_parse_date)
            # Remove rows with invalid dates
            df = df.dropna(subset=['filing_date'])
            years_coverage = (df['filing_date'].max() - df['filing_date'].min()).days / 365.25
            
            # Extract units from metadata if available
            units = set()
            if metadata and isinstance(metadata, dict):
                for key, value in metadata.items():
                    if isinstance(value, dict) and 'units' in value:
                        units.add(value['units'])
            
            return {
                'ticker': ticker,
                'missing_cols': missing_cols,
                'na_percentages': na_percentages,
                'years_coverage': years_coverage,
                'units': units,
                'file_size': os.path.getsize(tsv_path),
                'row_count': len(df),
                'col_count': len(df.columns)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing {tsv_path}: {str(e)}")
            return None

    def analyze_all_data(self):
        """Analyze all company files and aggregate results."""
        tsv_files, json_files = self.load_company_data()
        
        # Create progress bar
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("[green]Analyzing files...", total=len(tsv_files))
            
            for tsv_file in tsv_files:
                # Find corresponding metadata file
                metadata_file = None
                for json_file in json_files:
                    if json_file.stem.startswith(tsv_file.stem.split('_')[0]):
                        metadata_file = json_file
                        break
                
                result = self.analyze_file(tsv_file, metadata_file)
                if result:
                    # Update aggregated results
                    ticker = result['ticker']
                    self.results['missing_data'][ticker].extend(result['missing_cols'])
                    self.results['na_percentages'][ticker] = result['na_percentages']
                    self.results['units'][ticker].update(result['units'])
                    self.results['years_coverage'][ticker] = result['years_coverage']
                    
                    # Update concept stats
                    for col, na_pct in result['na_percentages'].items():
                        self.results['concept_stats'][col]['total_count'] += 1
                        if na_pct == 100:
                            self.results['concept_stats'][col]['missing_count'] += 1
                        elif na_pct > 0:
                            self.results['concept_stats'][col]['na_count'] += 1
                
                progress.update(task, advance=1)

    def generate_visualizations(self):
        """Create visualizations for the analysis results."""
        # Set style
        # Set default style
        plt.style.use('default')
        # Configure plot style manually
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.axisbelow'] = True
        
        # 1. Histogram of N/A percentages by ticker
        plt.figure(figsize=(12, 6))
        na_pcts = [np.mean(list(pcts.values())) for pcts in self.results['na_percentages'].values()]
        plt.hist(na_pcts, bins=50, edgecolor='black')
        plt.title('Distribution of N/A Percentages by Ticker')
        plt.xlabel('Average N/A Percentage')
        plt.ylabel('Number of Tickers')
        plt.savefig(self.output_dir / 'na_by_ticker.png')
        plt.close()
        
        # 2. Histogram of N/A percentages by concept
        plt.figure(figsize=(12, 6))
        concept_na_pcts = []
        for concept in self.results['concept_stats']:
            stats = self.results['concept_stats'][concept]
            if stats['total_count'] > 0:
                na_pct = ((stats['missing_count'] + stats['na_count']) / stats['total_count']) * 100
                concept_na_pcts.append(na_pct)
        
        plt.hist(concept_na_pcts, bins=50, edgecolor='black')
        plt.title('Distribution of N/A Percentages by Concept')
        plt.xlabel('N/A Percentage')
        plt.ylabel('Number of Concepts')
        plt.savefig(self.output_dir / 'na_by_concept.png')
        plt.close()
        
        # 3. Years coverage distribution
        plt.figure(figsize=(12, 6))
        plt.hist(list(self.results['years_coverage'].values()), bins=50, edgecolor='black')
        plt.title('Distribution of Years Coverage')
        plt.xlabel('Years of Data')
        plt.ylabel('Number of Tickers')
        plt.savefig(self.output_dir / 'years_coverage.png')
        plt.close()

    def generate_report(self):
        """Generate a comprehensive analysis report."""
        report = {
            'analysis_date': datetime.now().isoformat(),
            'total_tickers': len(self.results['missing_data']),
            'concept_summary': {},
            'ticker_summary': {},
            'units_summary': {ticker: list(units) for ticker, units in self.results['units'].items()},
            'years_coverage_summary': {
                'mean': np.mean(list(self.results['years_coverage'].values())),
                'median': np.median(list(self.results['years_coverage'].values())),
                'min': min(self.results['years_coverage'].values()),
                'max': max(self.results['years_coverage'].values())
            }
        }
        
        # Concept-level summary
        for concept, stats in self.results['concept_stats'].items():
            if stats['total_count'] > 0:
                report['concept_summary'][concept] = {
                    'missing_percentage': (stats['missing_count'] / stats['total_count']) * 100,
                    'partial_na_percentage': (stats['na_count'] / stats['total_count']) * 100,
                    'total_tickers': stats['total_count'],
                    'units': list(stats.get('units', []))
                }
        
        # Ticker-level summary
        for ticker, na_pcts in self.results['na_percentages'].items():
            report['ticker_summary'][ticker] = {
                'average_na_percentage': np.mean(list(na_pcts.values())),
                'missing_concepts': len(self.results['missing_data'][ticker]),
                'years_coverage': self.results['years_coverage'][ticker],
                'units': list(self.results['units'].get(ticker, []))
            }
        
        # Save report
        report_path = self.output_dir / 'analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate summary tables
        self._generate_summary_tables(report)
        
        return report

    def _generate_summary_tables(self, report: Dict):
        """Generate and save summary tables in CSV format."""
        # Concept summary table
        concept_df = pd.DataFrame.from_dict(report['concept_summary'], orient='index')
        concept_df.to_csv(self.output_dir / 'concept_summary.csv')
        
        # Ticker summary table
        ticker_df = pd.DataFrame.from_dict(report['ticker_summary'], orient='index')
        ticker_df.to_csv(self.output_dir / 'ticker_summary.csv')
        
        # Units summary table
        try:
            units_df = pd.DataFrame.from_dict(report['units_summary'], orient='index')
            units_df.to_csv(self.output_dir / 'units_summary.csv')
        except Exception as e:
            self.logger.error(f"Error saving units summary: {str(e)}")
            
        # Save the raw results as well for debugging
        with open(self.output_dir / 'raw_results.json', 'w') as f:
            json.dump({
                'missing_data': dict(self.results['missing_data']),
                'na_percentages': dict(self.results['na_percentages']),
                'units': {k: list(v) for k, v in self.results['units'].items()},
                'years_coverage': dict(self.results['years_coverage'])
            }, f, indent=2)

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        self.console.print("[bold blue]Starting SEC Data Analysis...[/bold blue]")
        
        # Run analysis
        self.analyze_all_data()
        self.console.print("[green]Data analysis complete.[/green]")
        
        # Generate visualizations
        self.generate_visualizations()
        self.console.print("[green]Visualizations generated.[/green]")
        
        # Generate report
        report = self.generate_report()
        self.console.print("[green]Analysis report generated.[/green]")
        
        # Print summary statistics
        self.console.print("\n[bold]Analysis Summary:[/bold]")
        self.console.print(f"Total tickers analyzed: {report['total_tickers']}")
        self.console.print(f"Average years of coverage: {report['years_coverage_summary']['mean']:.2f}")
        self.console.print(f"Output directory: {self.output_dir}")
        
        return report

def main():
    """Main execution function."""
    try:
        analyzer = SECDataAnalyzer()
        analyzer.run_analysis()
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
