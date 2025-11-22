#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--output_dir", default="metric_plots", 
                        help="Main directory to save plot subfolders")
    parser.add_argument("--metric", default="acc", choices=["acc", "rmse"],
                        help="Metric to plot (acc or rmse)")
    # New flag to control average plot
    parser.add_argument("--plot_average", action="store_true",
                        help="If set, plot a single average metric across all variables.")
    
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"❌ Error: Cannot find CSV file: {args.csv_file}")
        return
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # Check if metric column exists
    if args.metric not in df.columns:
        print(f"❌ Error: Metric '{args.metric}' not found in {args.csv_file}.")
        print(f"   Available columns: {list(df.columns)}")
        return

    df["Days"] = df["lead_time"] / 24.0
    df['level'] = df['level'].astype(str)

    # --- NEW LOGIC BRANCH ---
    if args.plot_average:
        # Plot a single average of the metric across all variables/levels
        print(f"📊 Plotting AVERAGE {args.metric.upper()} across all variables...")

        # Create the output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Calculate the mean of the metric, grouped by lead_time
        # This averages all variables and levels together for each lead time
        df_avg = df.groupby('lead_time')[args.metric].mean().reset_index()
        df_avg["Days"] = df_avg["lead_time"] / 24.0

        plt.figure(figsize=(10, 6))

        sns.lineplot(
            data=df_avg,
            x="Days",
            y=args.metric,
            markers=True,
            dashes=False,
            linewidth=2.5,
            label=f"Average {args.metric.upper()}"
        )
        
        metric_name = "RMSE" if args.metric == "rmse" else "ACC"
        plt.title(f"Average {metric_name} vs Lead Time", fontsize=16)
        plt.ylabel(metric_name, fontsize=12)
        plt.xlabel("Lead Time (Days)", fontsize=12)
        plt.grid(True, which="both", ls="--")

        # --- USER REQUEST: Add horizontal line at 0.5 for ACC ---
        if args.metric == "acc":
            plt.axhline(0.5, color='red', linestyle='--', label='ACC = 0.5')
            plt.ylim(0.0, 1.0) # Set Y-limit for ACC
        else:
            plt.ylim(bottom=0.0) # Set Y-limit for RMSE

        plt.legend() # Show legend (will include the 0.5 line)
        
        # Save plot in the main output_dir
        output_filename = os.path.join(args.output_dir, f"average_{args.metric}_plot.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Average plot saved to {output_filename}")

    else:
        # --- ORIGINAL LOGIC ---
        # Plot per-variable plots
        
        # Create a metric-specific subdirectory
        output_dir_metric = os.path.join(args.output_dir, args.metric)
        os.makedirs(output_dir_metric, exist_ok=True)
        
        variables = df['variable'].unique()
        print(f"📊 Plotting {args.metric.upper()} for each variable: {variables}")

        for var in variables:
            print(f"  ... Plotting {var}")
            plt.figure(figsize=(10, 6))
            
            df_var = df[df['variable'] == var].copy()
            
            if not pd.api.types.is_numeric_dtype(df_var[args.metric]) or \
               df_var[args.metric].isnull().all() or \
               (np.issubdtype(df_var[args.metric].dtype, np.number) and not np.isfinite(df_var[args.metric]).all()):
                
                print(f"  ... Warning: Data for metric '{args.metric}' on variable '{var}' is non-numeric, all-NaN, or contains Inf. Skipping.")
                plt.close()
                continue

            levels = df_var['level'].unique()

            sns.lineplot(
                data=df_var,
                x="Days",
                y=args.metric,
                hue="level",
                style="level",
                markers=True,
                dashes=False,
                linewidth=2.5
            )
            
            metric_name = "RMSE" if args.metric == "rmse" else "ACC"
            var_title = var.replace('_', ' ').title()
            
            plt.title(f"{metric_name} vs Lead Time for: {var_title}", fontsize=16)
            plt.ylabel(metric_name, fontsize=12)
            plt.xlabel("Lead Time (Days)", fontsize=12)
            plt.grid(True, which="both", ls="--")
            
            if args.metric == "acc":
                plt.ylim(0.0, 1.0)
            else:
                plt.ylim(bottom=0.0) 
                
            if len(levels) > 1:
                plt.legend(title="Level / Type")
            else:
                if plt.gca().get_legend():
                    plt.gca().get_legend().remove()

            safe_var_name = var.replace(" ", "_").replace("/", "_")
            output_filename = os.path.join(output_dir_metric, f"{args.metric}_plot_{safe_var_name}.png")
            
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"✅ All {args.metric.upper()} per-variable plots saved to {output_dir_metric}")

if __name__ == "__main__":
    main()