import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

class MarsHeatMapVisualizer:
    """
    Mock visualization of Mars Equipment Reliability Heat Map
    Simulates the C# implementation functionality in Python
    """
    
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.setup_data()
        self.setup_color_schemes()
        
        # Visualization state
        self.current_metric = 'reliability_score'
        self.current_instrument = 'all'
        self.seasonal_highlight = False
        self.highlighted_season = 0
        
        # Figure and axes
        self.fig = None
        self.ax_main = None
        self.scatter = None
        
    def setup_data(self):
        """Process and validate the loaded data"""
        print(f"Loaded {len(self.data)} heat map points")
        
        # Handle missing weather data (-999 sentinel values)
        weather_cols = ['temperature_celsius', 'temperature_range_celsius', 'wind_speed_avg_ms', 'wind_speed_max_ms']
        for col in weather_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].replace(-999, np.nan)
        
        # Get data bounds
        self.lat_bounds = (self.data['latitude'].min(), self.data['latitude'].max())
        self.lon_bounds = (self.data['longitude'].min(), self.data['longitude'].max())
        
        print(f"Coordinate bounds: Lat {self.lat_bounds}, Lon {self.lon_bounds}")
        print(f"Instruments: {sorted(self.data['instrument'].unique())}")
        print(f"Seasonal range: Ls {self.data['seasonal_longitude'].min()}° to {self.data['seasonal_longitude'].max()}°")
        
    def setup_color_schemes(self):
        """Define color schemes for different metrics using actual dataset columns"""
        self.color_schemes = {
            'reliability_score': {
                'cmap': 'RdYlGn',  # Red-Yellow-Green (poor to good)
                'label': 'Reliability Score',
                'vmin': 0, 'vmax': 1
            },
            'complexity_score': {
                'cmap': 'viridis',
                'label': 'Complexity Score', 
                'vmin': 0, 'vmax': 1
            },
            'temperature_celsius': {
                'cmap': 'coolwarm',
                'label': 'Temperature (°C)',
                'vmin': self.data['temperature_celsius'].replace(-999, np.nan).min(),
                'vmax': self.data['temperature_celsius'].replace(-999, np.nan).max()
            },
            'wind_speed_avg_ms': {
                'cmap': 'Blues',
                'label': 'Wind Speed (m/s)',
                'vmin': self.data['wind_speed_avg_ms'].replace(-999, np.nan).min(),
                'vmax': self.data['wind_speed_avg_ms'].replace(-999, np.nan).max()
            },
            'maintenance_activity_rate': {
                'cmap': 'Oranges',
                'label': 'Maintenance Rate',
                'vmin': 0, 'vmax': 1
            },
            'mission_age_factor': {
                'cmap': 'plasma',
                'label': 'Mission Age Factor',
                'vmin': 0, 'vmax': 3
            },
            'combined_thermal_stress': {
                'cmap': 'Reds',
                'label': 'Thermal Stress Combined',
                'vmin': self.data['combined_thermal_stress'].min(),
                'vmax': self.data['combined_thermal_stress'].max()
            },
            'success_rate': {
                'cmap': 'RdYlGn',
                'label': 'Success Rate',
                'vmin': 0, 'vmax': 1
            }
        }
        
        # Instrument colors for legends
        self.instrument_colors = {
            'IDA': '#FF6B6B',      # Red
            'HP3': '#4ECDC4',      # Teal  
            'SEIS': '#45B7D1',     # Blue
            'APSS': '#96CEB4',     # Green
            'CAMERA': '#FFEAA7',   # Yellow
            'WTS': '#DDA0DD'       # Purple
        }
    
    def filter_data(self):
        """Filter data based on current settings"""
        filtered_data = self.data.copy()
        
        # Filter by instrument
        if self.current_instrument != 'all':
            filtered_data = filtered_data[filtered_data['instrument'] == self.current_instrument]
        
        # Filter out NaN values for current metric if needed
        if self.current_metric in ['temperature_celsius', 'wind_speed_avg_ms', 'combined_thermal_stress']:
            filtered_data = filtered_data.dropna(subset=[self.current_metric])
        
        return filtered_data
    
    def create_interactive_plot(self):
        """Create the main interactive heat map visualization"""
        # Create figure with subplots for controls
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Mars Equipment Reliability Heat Map - Mock Visualization', 
                         fontsize=16, fontweight='bold')
        
        # Main heat map plot
        self.ax_main = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
        
        # Control panels
        ax_controls = plt.subplot2grid((4, 4), (0, 3), rowspan=4)
        ax_legend = plt.subplot2grid((4, 4), (3, 0), colspan=3)
        
        self.setup_main_plot()
        self.setup_controls(ax_controls)
        self.setup_legend(ax_legend)
        
        plt.tight_layout()
        plt.show()
        
    def setup_main_plot(self):
        """Setup the main heat map plot"""
        # Set Mars-like background
        self.ax_main.set_facecolor('#2F1B14')  # Dark reddish-brown
        
        # Draw coordinate grid
        self.draw_coordinate_grid()
        
        # Initial plot
        self.update_plot()
        
        # Set labels and limits
        self.ax_main.set_xlabel('Longitude (Mars Geographic)', fontsize=12)
        self.ax_main.set_ylabel('Latitude (Instrument Zones)', fontsize=12)
        self.ax_main.set_xlim(self.lon_bounds[0] - 10, self.lon_bounds[1] + 10)
        self.ax_main.set_ylim(self.lat_bounds[0] - 10, self.lat_bounds[1] + 10)
        
        # Add seasonal longitude annotations
        seasonal_positions = np.arange(-180, 181, 60)
        seasonal_labels = [f'Ls {((pos + 180) % 360):.0f}°' for pos in seasonal_positions]
        
        ax2 = self.ax_main.twiny()
        ax2.set_xlim(self.ax_main.get_xlim())
        ax2.set_xticks(seasonal_positions)
        ax2.set_xticklabels(seasonal_labels, fontsize=10)
        ax2.set_xlabel('Mars Seasonal Position (Solar Longitude)', fontsize=11)
        
    def draw_coordinate_grid(self):
        """Draw Mars coordinate grid"""
        # Latitude lines (every 30°)
        for lat in range(-60, 91, 30):
            self.ax_main.axhline(y=lat, color='white', alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Longitude lines (every 60°)  
        for lon in range(-180, 181, 60):
            self.ax_main.axvline(x=lon, color='white', alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Instrument zone boundaries
        zone_boundaries = [85, 60, 25, -5, -35, -60, -85]
        for boundary in zone_boundaries:
            self.ax_main.axhline(y=boundary, color='yellow', alpha=0.3, linestyle='--', linewidth=1)
            
    def update_plot(self):
        """Update the main plot with current settings"""
        filtered_data = self.filter_data()
        
        if len(filtered_data) == 0:
            if self.scatter:
                self.scatter.remove()
                self.scatter = None
            return
            
        # Get color scheme
        color_scheme = self.color_schemes[self.current_metric]
        
        # Point sizes based on sample count
        sizes = np.sqrt(filtered_data['sample_count']) * 20 + 30
        
        # Colors based on selected metric
        colors = filtered_data[self.current_metric]
        
        # Remove existing scatter plot
        if self.scatter:
            self.scatter.remove()
        
        # Create new scatter plot
        self.scatter = self.ax_main.scatter(
            filtered_data['longitude'], 
            filtered_data['latitude'],
            c=colors,
            s=sizes,
            cmap=color_scheme['cmap'],
            vmin=color_scheme['vmin'],
            vmax=color_scheme['vmax'],
            alpha=0.8,
            edgecolors='white',
            linewidth=0.5
        )
        
        # Add colorbar
        if hasattr(self, 'colorbar'):
            self.colorbar.remove()
        self.colorbar = plt.colorbar(self.scatter, ax=self.ax_main, shrink=0.8)
        self.colorbar.set_label(color_scheme['label'], fontsize=10)
        
        # Add seasonal highlight if enabled
        if self.seasonal_highlight:
            self.draw_seasonal_highlight(filtered_data)
            
        # Redraw
        self.fig.canvas.draw()
        
    def draw_seasonal_highlight(self, data):
        """Highlight current seasonal band"""
        season_lon_center = self.highlighted_season * 15 - 180  # Convert to longitude
        band_width = 30  # degrees
        
        # Draw highlight rectangle
        rect = Rectangle(
            (season_lon_center - band_width/2, self.lat_bounds[0] - 5),
            band_width,
            self.lat_bounds[1] - self.lat_bounds[0] + 10,
            facecolor='yellow',
            alpha=0.15,
            edgecolor='yellow',
            linewidth=2
        )
        self.ax_main.add_patch(rect)
        
    def setup_controls(self, ax):
        """Setup control panel"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Controls', ha='center', va='top', fontweight='bold', fontsize=14)
        
        # Metric selection
        ax.text(0.05, 0.85, 'Color Metric:', fontweight='bold', fontsize=11)
        metrics = list(self.color_schemes.keys())
        for i, metric in enumerate(metrics):
            y_pos = 0.80 - i * 0.04
            ax.text(0.1, y_pos, metric.replace('_', ' ').title(), fontsize=9)
            
        # Instrument selection  
        ax.text(0.05, 0.55, 'Instrument Filter:', fontweight='bold', fontsize=11)
        instruments = ['all'] + sorted(self.data['instrument'].unique())
        for i, instrument in enumerate(instruments):
            y_pos = 0.50 - i * 0.04
            ax.text(0.1, y_pos, instrument, fontsize=9)
            
        # Seasonal controls
        ax.text(0.05, 0.15, 'Seasonal Highlight:', fontweight='bold', fontsize=11)
        ax.text(0.1, 0.10, f'Season: Ls {self.highlighted_season * 15}°', fontsize=9)
        
        # Instructions
        ax.text(0.05, 0.02, 'Click legend items to interact', fontsize=8, style='italic')
        
    def setup_legend(self, ax):
        """Setup legend with instrument information"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Instrument legend
        ax.text(0.02, 0.9, 'Instrument Zones & Statistics:', fontweight='bold', fontsize=12)
        
        col_width = 0.16
        for i, instrument in enumerate(sorted(self.data['instrument'].unique())):
            col = i % 6
            x_pos = 0.02 + col * col_width
            
            inst_data = self.data[self.data['instrument'] == instrument]
            avg_reliability = inst_data['reliability_score'].mean()
            point_count = len(inst_data)
            
            # Instrument name with color
            color = self.instrument_colors.get(instrument, '#808080')
            ax.add_patch(plt.Circle((x_pos, 0.7), 0.01, color=color, alpha=0.8))
            ax.text(x_pos + 0.02, 0.7, f'{instrument}', fontweight='bold', fontsize=10)
            ax.text(x_pos + 0.02, 0.65, f'Rel: {avg_reliability:.3f}', fontsize=8)
            ax.text(x_pos + 0.02, 0.6, f'Points: {point_count}', fontsize=8)
            
        # Size legend
        ax.text(0.02, 0.45, 'Point Size = Sample Count', fontweight='bold', fontsize=10)
        
        # Sample size examples
        sample_sizes = [5, 15, 25]
        for i, size in enumerate(sample_sizes):
            x_pos = 0.1 + i * 0.15
            point_size = np.sqrt(size) * 20 + 30
            ax.scatter([x_pos], [0.35], s=[point_size], c='gray', alpha=0.7, edgecolors='black')
            ax.text(x_pos, 0.3, f'{size} samples', ha='center', fontsize=8)
            
        # Data summary
        ax.text(0.02, 0.15, f'Total Points: {len(self.data)} | Missing Weather: {self.data["temperature_celsius"].isna().sum()}', 
                fontsize=10)
        ax.text(0.02, 0.1, f'Sol Range: {self.data["sol_range_start"].min()}-{self.data["sol_range_end"].max()} | Seasons: {len(self.data["seasonal_bin"].unique())}', 
                fontsize=10)
        
    def create_analysis_plots(self):
        """Create additional analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Mars Equipment Reliability Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Reliability by Instrument
        ax1 = axes[0, 0]
        reliability_by_instrument = self.data.groupby('instrument')['reliability_score'].agg(['mean', 'std'])
        
        bars = ax1.bar(reliability_by_instrument.index, reliability_by_instrument['mean'], 
                      yerr=reliability_by_instrument['std'], capsize=5, alpha=0.8,
                      color=[self.instrument_colors.get(inst, '#808080') for inst in reliability_by_instrument.index])
        
        ax1.set_title('Average Reliability Score by Instrument')
        ax1.set_ylabel('Reliability Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Seasonal Reliability Patterns
        ax2 = axes[0, 1]
        seasonal_data = self.data.groupby('seasonal_bin')['reliability_score'].mean()
        seasonal_ls = seasonal_data.index * 15  # Convert to Ls degrees
        
        ax2.plot(seasonal_ls, seasonal_data.values, 'o-', linewidth=2, markersize=6, color='red')
        ax2.set_title('Reliability vs Mars Season')
        ax2.set_xlabel('Solar Longitude (Ls °)')
        ax2.set_ylabel('Average Reliability Score')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 360)
        
        # Plot 3: Temperature vs Reliability (excluding missing data)
        ax3 = axes[1, 0]
        temp_data = self.data.dropna(subset=['temperature_celsius'])
        
        if len(temp_data) > 0:
            scatter = ax3.scatter(temp_data['temperature_celsius'], temp_data['reliability_score'],
                                c=temp_data['combined_thermal_stress'], cmap='Reds', alpha=0.6, s=50)
            ax3.set_title('Temperature vs Reliability')
            ax3.set_xlabel('Temperature (°C)')
            ax3.set_ylabel('Reliability Score')
            plt.colorbar(scatter, ax=ax3, label='Thermal Stress Combined', shrink=0.8)
        else:
            ax3.text(0.5, 0.5, 'No Temperature Data Available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Temperature vs Reliability (No Data)')
        
        # Plot 4: Mission Age Impact
        ax4 = axes[1, 1]
        age_bins = pd.cut(self.data['mission_age_factor'], bins=5)
        age_reliability = self.data.groupby(age_bins)['reliability_score'].agg(['mean', 'count'])
        
        bars = ax4.bar(range(len(age_reliability)), age_reliability['mean'], 
                      alpha=0.8, color='orange')
        ax4.set_title('Reliability vs Mission Age')
        ax4.set_xlabel('Mission Age Factor (binned)')
        ax4.set_ylabel('Average Reliability Score')
        ax4.set_xticks(range(len(age_reliability)))
        ax4.set_xticklabels([f'{interval.left:.1f}-{interval.right:.1f}' for interval in age_reliability.index], 
                           rotation=45)
        
        # Add count labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={age_reliability.iloc[i]["count"]}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
    def generate_summary_report(self):
        """Generate text summary of the heat map data"""
        print("\n" + "="*60)
        print("MARS EQUIPMENT RELIABILITY HEAT MAP SUMMARY")
        print("="*60)
        
        # Overall statistics
        print(f"\nOVERALL STATISTICS:")
        print(f"Total heat map points: {len(self.data)}")
        print(f"Instruments: {len(self.data['instrument'].unique())}")
        print(f"Seasonal coverage: {len(self.data['seasonal_bin'].unique())} seasons")
        print(f"Sol range: {self.data['sol_range_start'].min()} to {self.data['sol_range_end'].max()}")
        
        # Reliability statistics by instrument
        print(f"\nRELIABILITY BY INSTRUMENT:")
        for instrument in sorted(self.data['instrument'].unique()):
            inst_data = self.data[self.data['instrument'] == instrument]
            print(f"{instrument:8}: {inst_data['reliability_score'].mean():.3f} ± {inst_data['reliability_score'].std():.3f} ({len(inst_data)} points)")
        
        # Environmental data coverage
        weather_coverage = {
            'temperature': (~self.data['temperature_celsius'].isna()).sum(),
            'wind_speed_avg': (~self.data['wind_speed_avg_ms'].isna()).sum(),
            'wind_speed_max': (~self.data['wind_speed_max_ms'].isna()).sum(),
            'temp_range': (~self.data['temperature_range_celsius'].isna()).sum()
        }
        
        print(f"\nWEATHER DATA COVERAGE:")
        for metric, count in weather_coverage.items():
            coverage_pct = (count / len(self.data)) * 100
            print(f"{metric:15}: {count:4d} points ({coverage_pct:5.1f}% coverage)")
        
        # Seasonal patterns
        print(f"\nSEASONAL RELIABILITY PATTERNS:")
        seasonal_stats = self.data.groupby('seasonal_bin')['reliability_score'].agg(['mean', 'count'])
        best_season = seasonal_stats['mean'].idxmax()
        worst_season = seasonal_stats['mean'].idxmin()
        
        print(f"Best season:  Ls {best_season * 15:3.0f}° (Reliability: {seasonal_stats.loc[best_season, 'mean']:.3f})")
        print(f"Worst season: Ls {worst_season * 15:3.0f}° (Reliability: {seasonal_stats.loc[worst_season, 'mean']:.3f})")
        
        # Activity patterns
        print(f"\nACTIVITY PATTERNS:")
        print(f"Science activity rate:     {self.data['science_activity_rate'].mean():.3f}")
        print(f"Maintenance activity rate: {self.data['maintenance_activity_rate'].mean():.3f}")
        print(f"Recovery activity rate:    {self.data['recovery_activity_rate'].mean():.3f}")
        print(f"Power intensive rate:      {self.data['power_intensive_rate'].mean():.3f}")

def create_sample_data():
    """
    Create sample data for testing if no CSV exists
    This simulates the output from the previous processing script
    """
    np.random.seed(42)  # For reproducible results
    
    instruments = ['IDA', 'HP3', 'SEIS', 'APSS', 'CAMERA', 'WTS']
    seasonal_bins = range(24)  # 24 seasonal positions
    
    data = []
    
    # Instrument zone mapping (same as processing script)
    instrument_zones = {
        'IDA': {'lat_base': 45, 'lat_range': 25},
        'HP3': {'lat_base': 15, 'lat_range': 20},
        'SEIS': {'lat_base': -15, 'lat_range': 20},
        'APSS': {'lat_base': -45, 'lat_range': 25},
        'CAMERA': {'lat_base': 70, 'lat_range': 15},
        'WTS': {'lat_base': -70, 'lat_range': 15}
    }
    
    for instrument in instruments:
        zone = instrument_zones[instrument]
        
        for season_bin in seasonal_bins:
            seasonal_lon = season_bin * 15 + 7.5
            geo_longitude = ((seasonal_lon + 180) % 360) - 180
            
            # Simulate instrument-specific performance patterns
            if instrument == 'HP3':
                # HP3 had known issues
                base_reliability = 0.65
                seasonal_variation = 0.2 * np.sin(np.radians(seasonal_lon))
            elif instrument == 'SEIS':
                # SEIS was very reliable
                base_reliability = 0.90
                seasonal_variation = 0.05 * np.sin(np.radians(seasonal_lon + 90))
            elif instrument == 'IDA':
                # IDA moderate with thermal sensitivity
                base_reliability = 0.82
                seasonal_variation = 0.15 * np.cos(np.radians(seasonal_lon))
            else:
                # Other instruments
                base_reliability = 0.78
                seasonal_variation = 0.1 * np.sin(np.radians(seasonal_lon + np.random.uniform(0, 360)))
            
            reliability = np.clip(base_reliability + seasonal_variation + np.random.normal(0, 0.05), 0.3, 0.98)
            
            # Generate latitude with some spread
            lat_offset = np.random.uniform(-zone['lat_range']/3, zone['lat_range']/3)
            latitude = np.clip(zone['lat_base'] + lat_offset, -85, 85)
            
            data.append({
                'latitude': round(latitude, 3),
                'longitude': round(geo_longitude, 3),
                'instrument': instrument,
                'seasonal_longitude': seasonal_lon,
                'seasonal_bin': season_bin,
                'sol_range_start': season_bin * 60 + np.random.randint(1, 10),
                'sol_range_end': (season_bin + 1) * 60 + np.random.randint(40, 50),
                'median_sol': season_bin * 60 + 30,
                'reliability_score': round(reliability, 4),
                'power_reliability': round(np.clip(reliability + np.random.normal(0, 0.03), 0, 1), 4),
                'routine_ratio': round(np.random.uniform(0.6, 0.9), 4),
                'complexity_score': round(np.random.uniform(0.2, 0.8), 4),
                'complexity_std': round(np.random.uniform(0.05, 0.15), 4),
                'sample_count': np.random.randint(5, 25),
                'reliability_std': round(np.random.uniform(0.02, 0.12), 4),
                'mission_age_factor': round(season_bin / 8, 2),  # Increases with time
                'systems_involved': round(np.random.uniform(1, 4), 1),
                'operational_tempo': round(np.random.uniform(0.8, 2.2), 2),
                'science_activity_rate': round(np.random.uniform(0.4, 0.8), 4),
                'maintenance_activity_rate': round(np.random.uniform(0.1, 0.4), 4),
                'recovery_activity_rate': round(np.random.uniform(0.0, 0.2), 4),
                'movement_required_rate': round(np.random.uniform(0.3, 0.7), 4),
                'power_intensive_rate': round(np.random.uniform(0.2, 0.6), 4),
                'temperature_celsius': round(np.random.uniform(-85, -50), 1) if np.random.random() > 0.3 else -999,
                'wind_speed_variability': round(np.random.uniform(0.1, 1.5), 2),
                'weather_data_quality': round(np.random.uniform(5, 20), 0),
                
                # Thermal stress factors  
                'seasonal_intensity_factor': round(np.random.uniform(0.2, 1.8), 2),
                'orbital_thermal_factor': round(np.random.uniform(0.1, 1.2), 2),
                'diurnal_thermal_factor': round(np.random.uniform(0.3, 1.5), 2),
                'combined_thermal_stress': round(np.random.uniform(0.5, 2.5), 2) if np.random.random() > 0.3 else -999
            })
    
    return pd.DataFrame(data)

def main():
    """Main execution function"""
    csv_path = r'data-analysis-toolkit\r_style_analysis_framework\python_analysis_framework\02_Data_Cleaning_and_Preparation\Prepped_data\mars_heat_map_coordinates.csv'
    
    # Try to load existing CSV, create sample data if not found
    try:
        print(f"Loading heat map data from {csv_path}...")
        visualizer = MarsHeatMapVisualizer(csv_path)
    except FileNotFoundError:
        print(f"CSV file {csv_path} not found. Creating sample data...")
        sample_data = create_sample_data()
        sample_data.to_csv(csv_path, index=False)
        print(f"Sample data saved to {csv_path}")
        visualizer = MarsHeatMapVisualizer(csv_path)
    
    # Generate summary report
    visualizer.generate_summary_report()
    
    print("\nCreating visualizations...")
    
    # Create analysis plots first
    visualizer.create_analysis_plots()
    
    # Create interactive heat map (note: interactivity limited in matplotlib)
    visualizer.create_interactive_plot()
    
    print("\nVisualization complete!")
    print("The C# implementation would provide real-time interactivity")
    print("including metric switching, instrument filtering, and seasonal animation.")

if __name__ == "__main__":
    main()
