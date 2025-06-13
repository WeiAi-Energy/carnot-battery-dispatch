import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Union, Tuple


def load_price_data(csv_path: str) -> pd.DataFrame:
    """
    Load price data from CSV, keeping the original row structure.
    Each 24 rows counts as one day, regardless of time format.

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with loaded price data
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Ensure we have the expected columns
    required_cols = ['Date', 'Time', 'LMP ($/MWh)']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the CSV file")

    # Add a sequential hour index starting from 0
    df['hour_idx'] = range(len(df))

    # Calculate day index (integer division by 24)
    df['day_idx'] = df['hour_idx'] // 24

    return df


def find_representative_periods(price_data: pd.DataFrame,
                                price_col: str = 'LMP ($/MWh)',
                                week_length: int = 7 * 24,
                                month_length: int = 30 * 24,
                                time_step: int = 24) -> Dict[str, Any]:
    """
    Find the most representative periods based on statistical similarity to the full dataset.

    Args:
        price_data: DataFrame with price data
        price_col: Column name for price data
        week_length: Length of a week in hours
        month_length: Length of multiple weeks in hours
        time_step: Step size for the start of periods (24 for daily alignment)

    Returns:
        Dictionary with representative periods and statistics
    """
    # Extract price data
    prices = price_data[price_col].values

    # Calculate global statistics
    global_mean = np.mean(prices)
    global_std = np.std(prices)

    # Find best single week
    best_week = find_best_period(prices, week_length, time_step, global_mean, global_std)

    # Find best consecutive 7 weeks
    best_month = find_best_period(prices, month_length, time_step, global_mean, global_std)

    # Extract periods from the original DataFrame
    best_week_data = price_data.iloc[best_week['start_idx']:best_week['end_idx']]
    best_month_data = price_data.iloc[best_month['start_idx']:best_month['end_idx']]

    # Prepare results
    results = {
        'best_week_data': best_week_data,
        'best_month_data': best_month_data,
        'best_week_info': best_week,
        'best_month_info': best_month,
        'global_stats': {
            'mean': global_mean,
            'std': global_std
        }
    }

    return results


def find_best_period(prices: np.ndarray, period_length: int, time_step: int,
                     global_mean: float, global_std: float) -> Dict[str, Any]:
    """
    Find the period with mean and standard deviation closest to global statistics.

    Args:
        prices: Array of price data
        period_length: Length of period to find in hours
        time_step: Step size for starting points
        global_mean: Mean of the entire dataset
        global_std: Standard deviation of the entire dataset

    Returns:
        Dictionary with information about the best period
    """
    n_hours = len(prices)

    # Ensure period_length is valid
    if period_length > n_hours:
        raise ValueError(f"Period length ({period_length}) must be less than timeseries length ({n_hours})")

    # Initialize tracking variables
    best_score = float('inf')
    best_start_idx = 0
    best_period_mean = 0
    best_period_std = 0

    # Iterate through possible periods (starting at time_step intervals)
    for start_idx in range(0, n_hours - period_length + 1, time_step):
        # Extract period data
        period_data = prices[start_idx:start_idx + period_length]

        # Calculate period statistics
        period_mean = np.mean(period_data)
        period_std = np.std(period_data)

        # Calculate normalized differences
        mean_diff = abs(period_mean - global_mean) / global_mean
        std_diff = abs(period_std - global_std) / global_std

        # Combined score (lower is better)
        score = mean_diff + std_diff * 2.5
        if score < best_score:
            best_score = score
            best_start_idx = start_idx
            best_period_mean = period_mean
            best_period_std = period_std

    # Prepare result
    result = {
        'start_idx': best_start_idx,
        'end_idx': best_start_idx + period_length + 24,
        'start_day': best_start_idx // 24,
        'end_day': (best_start_idx + period_length) // 24,
        'length_hours': period_length,
        'length_days': period_length / 24,
        'score': best_score,
        'mean': best_period_mean,
        'std': best_period_std,
        'mean_diff_pct': abs(best_period_mean - global_mean) / global_mean * 100,
        'std_diff_pct': abs(best_period_std - global_std) / global_std * 100
    }

    return result


def visualize_results(price_data: pd.DataFrame, results: Dict[str, Any], price_col: str = 'LMP ($/MWh)') -> None:
    """
    Visualize the representative periods and compare statistics.

    Args:
        price_data: Original price DataFrame
        results: Results from find_representative_periods
        price_col: Column name for the price data
    """
    # Create figure
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=False)

    # First subplot: Full year data (no changes)
    x_full = price_data['hour_idx'].values
    axes[0].plot(x_full, price_data[price_col], '-', alpha=0.6)
    axes[0].set_title('Full Year')
    axes[0].set_ylabel('LMP ($/MWh)')
    axes[0].set_xlabel('Hour Index')
    axes[0].grid(True)
    axes[0].axvspan(results['best_week_info']['start_idx'],
                    results['best_week_info']['end_idx'],
                    color='r', alpha=0.2, label='Best Week + One Extra Day')
    # axes[0].axvspan(results['best_month_info']['start_idx'],
    #                 results['best_month_info']['end_idx'],
    #                 color='g', alpha=0.2, label='Best Month + One Extra Day')
    axes[0].legend(loc='upper left')
    axes[0].set_xlim(0, 8760)

    # Second subplot: Best week data with proper hour ticks
    best_week = results['best_week_data']
    # Get start and end hours
    week_start_hour = results['best_week_info']['start_idx']
    week_end_hour = results['best_week_info']['end_idx']
    # Use actual hour indices instead of dataframe indices
    x_hours = np.arange(week_start_hour, week_end_hour)

    axes[1].plot(x_hours, best_week[price_col], '-', label='Best Week + One Extra Day')
    axes[1].set_title(
        f'Best Week + One Extra Day (Days {results["best_week_info"]["start_day"]}–{results["best_week_info"]["end_day"]})')
    axes[1].set_ylabel('LMP ($/MWh)')
    axes[1].set_xlabel('Hour Index')
    axes[1].grid(True)

    # Set x-axis ticks at 24-hour intervals
    ticks_week = np.arange(week_start_hour, week_end_hour + 1, 24)
    axes[1].set_xticks(ticks_week)
    # Limit x-axis range to exact start and end hours
    axes[1].set_xlim(week_start_hour, week_end_hour)

    # Third subplot: Best month data with proper hour ticks
    best_multi = results['best_month_data']
    # Get start and end hours
    month_start_hour = results['best_month_info']['start_idx']
    month_end_hour = results['best_month_info']['end_idx']
    # Use actual hour indices instead of dataframe indices
    x_hours_month = np.arange(month_start_hour, month_end_hour)

    axes[2].plot(x_hours_month, best_multi[price_col], '-', label='Best Month + One Extra Day')
    axes[2].set_title(
        f'Best Month + One Extra Day (Days {results["best_month_info"]["start_day"]}–{results["best_month_info"]["end_day"]})')
    axes[2].set_ylabel('LMP ($/MWh)')
    axes[2].set_xlabel('Hour Index')
    axes[2].grid(True)

    # Set x-axis ticks at 24-hour intervals
    ticks_month = np.arange(month_start_hour, month_end_hour + 1, 24*5)
    axes[2].set_xticks(ticks_month)
    # Limit x-axis range to exact start and end hours
    axes[2].set_xlim(month_start_hour, month_end_hour)

    # Statistics information
    gw = results['global_stats']
    w = results['best_week_info']
    mw = results['best_month_info']

    stats_text = (
        f"Full Year: μ = {gw['mean']:.2f}, σ = {gw['std']:.2f}\n"
        f"Best Week + One Extra Day: μ = {w['mean']:.2f} ({w['mean_diff_pct']:.1f}%), "
        f"σ = {w['std']:.2f} ({w['std_diff_pct']:.1f}%)\n"
        f"Best Month + One Extra Day: μ = {mw['mean']:.2f} "
        f"({mw['mean_diff_pct']:.1f}%), σ = {mw['std']:.2f} ({mw['std_diff_pct']:.1f}%)"
    )

    fig.text(0.5, 0.0, stats_text,
             ha='center', va='bottom',
             fontsize=18,
             linespacing=1.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, hspace=0.22, top=0.97)
    plt.show()
    plt.savefig('representative week.png',dpi=600)


def extract_to_csv(results: Dict[str, Any], price_data: pd.DataFrame, output_prefix: str = 'representative',
                   extra_days: int = 1) -> Tuple[str, str]:
    """
    Extract the representative periods to CSV files, including extra days for prediction.

    Args:
        results: Results from find_representative_periods
        price_data: Full price DataFrame
        output_prefix: Prefix for output filenames
        extra_days: Number of extra days to include

    Returns:
        Tuple of filenames for the saved CSV files
    """
    # Calculate extra hours
    extra_hours = extra_days * 24

    # For best week, include extra day
    week_start_idx = results['best_week_info']['start_idx']
    week_end_idx = results['best_week_info']['end_idx']

    # Make sure we don't go out of bounds
    if week_end_idx > len(price_data):
        # Handle case where we need to wrap around
        week_part1 = price_data.iloc[week_start_idx:len(price_data)]
        week_part2 = price_data.iloc[0:(week_end_idx - len(price_data))]
        best_week_data_extended = pd.concat([week_part1, week_part2])
    else:
        best_week_data_extended = price_data.iloc[week_start_idx:week_end_idx]

    # For best multi-week, include extra day
    month_start_idx = results['best_month_info']['start_idx']
    month_end_idx = results['best_month_info']['end_idx']

    # Make sure we don't go out of bounds
    if month_end_idx > len(price_data):
        # Handle case where we need to wrap around
        month_part1 = price_data.iloc[month_start_idx:len(price_data)]
        month_part2 = price_data.iloc[0:(month_end_idx - len(price_data))]
        best_month_data_extended = pd.concat([month_part1, month_part2])
    else:
        best_month_data_extended = price_data.iloc[month_start_idx:month_end_idx]

    # Add information about the original and extended periods
    week_days = results['best_week_info']['length_days']
    month_days = results['best_month_info']['length_days']

    # Save best week with extra day
    week_filename = f"{output_prefix}_week.csv"
    best_week_data_extended.to_csv(week_filename, index=False)

    # Save best multi-week with extra day
    month_filename = f"{output_prefix}_month.csv"
    best_month_data_extended.to_csv(month_filename, index=False)

    print(f"Saved representative week to {week_filename}")
    print(f"Saved representative month to {month_filename}")

    return week_filename, month_filename


def main(csv_path: str, output_prefix: str = 'representative', extra_days: int = 1):
    """
    Main function to find and visualize representative periods.

    Args:
        csv_path: Path to the CSV file
        output_prefix: Prefix for output filenames
        extra_days: Number of extra days to include in CSV output
    """
    # Load data
    print(f"Loading data from {csv_path}...")
    price_data = load_price_data(csv_path)

    # Find representative periods
    print("Finding representative periods...")
    results = find_representative_periods(price_data)

    # Print summary
    week_info = results['best_week_info']
    month_info = results['best_month_info']

    print("\nResults Summary:")
    print(f"Full Dataset: Mean = {results['global_stats']['mean']:.2f}, Std = {results['global_stats']['std']:.2f}")
    print(f"\nBest Representative Week:")
    print(f"  Start Day: {week_info['start_day']} (Hour {week_info['start_idx']})")
    print(f"  End Day: {week_info['end_day']} (Hour {week_info['end_idx']})")
    print(f"  Mean: {week_info['mean']:.2f} (Difference: {week_info['mean_diff_pct']:.2f}%)")
    print(f"  Std: {week_info['std']:.2f} (Difference: {week_info['std_diff_pct']:.2f}%)")

    print(f"\nBest Representative 7 Weeks:")
    print(f"  Start Day: {month_info['start_day']} (Hour {month_info['start_idx']})")
    print(f"  End Day: {month_info['end_day']} (Hour {month_info['end_idx']})")
    print(f"  Mean: {month_info['mean']:.2f} (Difference: {month_info['mean_diff_pct']:.2f}%)")
    print(f"  Std: {month_info['std']:.2f} (Difference: {month_info['std_diff_pct']:.2f}%)")

    # Visualize results
    print("\nGenerating visualization...")
    visualize_results(price_data, results)

    # Save to CSV with extra day
    print(f"\nSaving representative periods to CSV with {extra_days} extra day(s)...")
    week_file, month_file = extract_to_csv(results, price_data, output_prefix, extra_days)

    print("\nDone!")
    return results


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Find representative periods in CAISO price data')
    parser.add_argument('csv_path', nargs='?', default="CAISO2024.csv", help='Path to the CSV file')
    parser.add_argument('--output-prefix', '-o', default="representative", help='Prefix for output filenames')
    parser.add_argument('--extra-days', '-e', type=int, default=1, help='Number of extra days to include in CSV output')

    args = parser.parse_args()

    main(args.csv_path, args.output_prefix, args.extra_days)