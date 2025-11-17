# -*- coding: utf-8 -*-
"""
plot_tom_moving_average.py
Generate moving average plot for Theory of Mind (ToM) ratings
Scale: -10 to +10
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Font settings for better compatibility
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_tom_data(csv_file):
    """Load ToM data from CSV"""
    df = pd.read_csv(csv_file)
    
    # Verify required columns exist
    required_cols = ['frame_index', 'timestamp_sec', 'tom_rating']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    return df[required_cols]


def plot_tom_moving_average(df, window=10, output_file='tom_moving_average.png', dpi=300):
    """
    Plot Theory of Mind moving average with -10 to +10 scale
    
    Parameters:
    -----------
    df : DataFrame
        Data with columns: frame_index, timestamp_sec, tom_rating
    window : int
        Moving average window size (number of frames)
    output_file : str
        Output filename
    dpi : int
        Image resolution
    """
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Calculate moving average
    tom_ma = df['tom_rating'].rolling(window=window, center=True).mean()
    time_sec = df['timestamp_sec']  # Use seconds directly
    
    # Plot raw data (semi-transparent)
    ax.plot(time_sec, df['tom_rating'], color='#2E86AB', alpha=0.25, 
            linewidth=1.5, label='Raw frame data', zorder=1)
    
    # Plot moving average (bold line)
    ax.plot(time_sec, tom_ma, color='#2E86AB', linewidth=3.5, 
            label=f'{window}-frame moving average', zorder=2)
    
    # Fill area under curve for better visibility
    # Positive area (above 0)
    ax.fill_between(time_sec, 0, tom_ma, 
                     where=(tom_ma >= 0), color='#4CAF50', alpha=0.2, 
                     interpolate=True, zorder=0)
    
    # Negative area (below 0)
    ax.fill_between(time_sec, 0, tom_ma, 
                     where=(tom_ma < 0), color='#F44336', alpha=0.2, 
                     interpolate=True, zorder=0)
    
    # Zero reference line (neutral baseline)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.6, 
               label='Neutral baseline', zorder=1)
    
    # Labels and title
    ax.set_xlabel('Time (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Theory of Mind Rating (-10 to +10)', fontsize=13, fontweight='bold')
    ax.set_ylim(-11, 11)
    ax.set_title(f'Theory of Mind: {window}-Frame Moving Average', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Calculate statistics (for printing only, not displayed on plot)
    tom_mean = df['tom_rating'].mean()
    tom_median = df['tom_rating'].median()
    tom_std = df['tom_rating'].std()
    
    positive_count = (df['tom_rating'] > 0).sum()
    negative_count = (df['tom_rating'] < 0).sum()
    neutral_count = (df['tom_rating'] == 0).sum()
    total_count = len(df)
    
    positive_pct = positive_count / total_count * 100
    negative_pct = negative_count / total_count * 100
    neutral_pct = neutral_count / total_count * 100
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    # Print summary
    print(f"\n📊 Statistics Summary:")
    print(f"  Total frames: {total_count}")
    print(f"  Mean: {tom_mean:+.2f}")
    print(f"  Median: {tom_median:+.2f}")
    print(f"  Std Dev: {tom_std:.2f}")
    print(f"  Range: [{df['tom_rating'].min():+}, {df['tom_rating'].max():+}]")
    print(f"  Positive: {positive_pct:.1f}%")
    print(f"  Negative: {negative_pct:.1f}%")
    print(f"  Neutral: {neutral_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Theory of Mind Moving Average Plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_tom_moving_average.py --input tom_results.csv --output tom_plot.png
  python plot_tom_moving_average.py --input tom_results.csv --window 15
  python plot_tom_moving_average.py --input tom_results.csv --window 5 --dpi 600
        """
    )
    
    parser.add_argument("--input", required=True, 
                       help="Input CSV file with ToM ratings")
    parser.add_argument("--output", default="tom_moving_average.png",
                       help="Output image file (default: tom_moving_average.png)")
    parser.add_argument("--window", type=int, default=10,
                       help="Moving average window size in frames (default: 10)")
    parser.add_argument("--dpi", type=int, default=300,
                       help="Image resolution (default: 300)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return
    
    # Validate parameters
    if args.window < 1:
        print(f"❌ Error: Window size must be >= 1")
        return
    
    if args.dpi < 72:
        print(f"❌ Error: DPI must be >= 72")
        return
    
    print("=" * 80)
    print("📊 THEORY OF MIND - MOVING AVERAGE PLOT")
    print("=" * 80)
    print(f"\n📂 Input: {args.input}")
    print(f"📊 Output: {args.output}")
    print(f"🔢 Window size: {args.window} frames")
    print(f"🎨 Resolution: {args.dpi} DPI")
    
    # Load data
    print(f"\n📥 Loading data...")
    df = load_tom_data(args.input)
    print(f"✓ Loaded {len(df)} frames")
    
    # Generate plot
    print(f"\n🎨 Generating plot...")
    plot_tom_moving_average(df, args.window, args.output, args.dpi)
    
    print("\n" + "=" * 80)
    print("✓✓✓ COMPLETE! ✓✓✓")
    print("=" * 80)


if __name__ == "__main__":
    main()
