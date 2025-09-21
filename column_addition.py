import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
import re

warnings.filterwarnings('ignore')


class MarsMasterDatasetCreator:
    """
    Builds a sol-granular Mars InSight master dataset from:
      Weather (15-min bins):
        - bin (datetime), BMY_BASE_ROD_TEMP, BMY_MID_ROD_TEMP, BMY_TIP_ROD_TEMP,
          BMY_HORIZONTAL_WIND_SPEED, BMY_WIND_DIRECTION, file_id, file_id_str, solar_longitude
      Activities (daily rows):
        - datetime, sol, end, look_ahead, activity, solar_longitude_deg, date, year, month,
          day, hour, day_of_year, weekday, operational_status, criticality, power_status
    Emphasizes Mars-correct constants, vectorized processing, and powerful compound variables.
    """

    def __init__(self):
        # Mission constants (Mars-correct)
        self.MARS_SOL_TO_EARTH_DAYS = 1.02749          # 1 sol in Earth days
        self.MARS_YEAR_SOLS = 669                       # ~1 Mars year in sols
        self.DUST_SEASON_LS = (180, 300)                # Ls dust-prone season
        self.PERIHELION_LS = 250                        # Approx. perihelion
        self.LANDING_EARTH_DT = pd.Timestamp('2018-11-26 19:52:59', tz='UTC')  # InSight landing

        # Instrument keyword lexicon (InSight focused)
        self.instrument_keywords = {
            'IDA': ['ida', 'arm', 'idc', 'icc', 'grapple', 'scoop', 'unstow', 'stow', 'place'],
            'HP3': ['hp3', 'mole', 'hammer', 'hammering', 'pinning', 'backout', 'thermal probe', 'rad'],
            'SEIS': ['seis', 'vbb', 'sp', 'seismometer', 'tilt', 'level', 'levelling', 'vibration', 'boom'],
            'APSS': ['apss', 'twins', 'wind', 'pressure', 'barometer', 'temperature', 'met', 'weather'],
            'CAMERA': ['imaging', 'image', 'mosaic', 'panorama', 'photo', 'idc', 'icc'],
            'WTS': ['wts', 'wind', 'thermal', 'shield']
        }

        # Activity type keywords (reduced ambiguity)
        self.activity_types = {
            'SCIENCE': ['imaging', 'observation', 'survey', 'seismic', 'recording', 'monitoring', 'measure'],
            'MAINTENANCE': ['calibration', 'characterization', 'checkout', 'maintenance', 'leveling', 'tuning'],
            'DEPLOYMENT': ['deploy', 'unstow', 'stow', 'place', 'position', 'release', 'capture', 'adjust', 'install'],
            'RECOVERY': ['recovery', 'safe mode', 'safed', 'unsafed', 'repair', 'troubleshoot', 'anomaly', 'backout', 'fault'],
            'THERMAL': ['thermal', 'heater', 'heating', 'cooldown', 'temperature'],
            'COMMUNICATION': ['uplink', 'downlink', 'relay', 'x band', 'update', 'wutt']
        }
    
    def create_master_dataset(self,
                              activities_file='mars_activities_dataframe.csv',
                              weather_file='mars_weather_15min_with_solar_longitude.csv'):
        # Load
        activities_df = pd.read_csv(activities_file)
        weather_df = pd.read_csv(weather_file)

        # Build sol-level weather and activities
        weather_base = self._create_weather_base(weather_df)
        activities_agg = self._aggregate_activities(activities_df)

        # Merge
        master_df = weather_base.merge(activities_agg, on='sol', how='outer').sort_values('sol').reset_index(drop=True)

        # Normalize types and fill
        master_df['sol'] = master_df['sol'].astype('int32')
        if 'has_activity' in master_df.columns:
            master_df['has_activity'] = master_df['has_activity'].fillna(0).astype('int8')
        if 'activity_count' in master_df.columns:
            master_df['activity_count'] = master_df['activity_count'].fillna(0).astype('int16')
        if 'activity_text' in master_df.columns:
            master_df['activity_text'] = master_df['activity_text'].fillna('')

        # Fill/derive Ls robustly
        master_df['solar_longitude_deg'] = self._unify_and_fill_ls(master_df)

        # NEW: impute weather features for sols that only have activities
        master_df = self._impute_after_merge(master_df)

        # Temporal and mission context
        master_df = self._add_sol_temporal_features(master_df)
        master_df = self._add_mission_context(master_df)

        # Activity analysis
        master_df = self._add_activity_analysis(master_df)

        # Environmental features
        master_df = self._add_environmental_features(master_df)

        # Operational features
        master_df = self._add_operational_features(master_df)

        # Dust storm features
        master_df = self._add_dust_storm_features(master_df)

        # Equipment features
        master_df = self._add_equipment_features(master_df)

        # Powerful compound variables
        master_df = self._add_compound_variables(master_df)

        # Helpful aliases to maximize valid columns
        master_df['solar_longitude'] = master_df['solar_longitude_deg'].astype('float32')
        if 'temp_avg' in master_df.columns:
            master_df['avg_temp'] = master_df['temp_avg'].astype('float32')

        # Optimize and return
        master_df = self._optimize_dataset(master_df)
        return master_df

    # ============== NEW: Post-merge imputation method ==============
    
    def _impute_after_merge(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create 30° Ls bins for season-aware imputation
        df['ls_bin'] = (df['solar_longitude_deg'] // 30 * 30).astype('float32')

        # Weather summary columns to impute when weather is missing
        weather_cols = [
            'temp_min','temp_max','temp_avg','temp_p05','temp_p95','temp_range',
            'wind_speed_avg','wind_speed_max','wind_speed_std','wind_speed_p95',
            'wind_gust_factor','wind_gustiness','wind_direction_avg','wind_direction_std',
            'weather_readings_count','weather_data_completeness','weather_available_today'
        ]
        for col in weather_cols:
            if col in df.columns:
                # Prefer seasonal (Ls-bin) median where available
                df[col] = df[col].fillna(df.groupby('ls_bin')[col].transform('median'))
                # Nearest-by-sol fill, then global median as last resort
                df[col] = df[col].ffill().bfill()
                if df[col].isna().any():
                    med = df[col].median()
                    df[col] = df[col].fillna(med)

        # Bridge activity avg_temp with weather temp_avg when activity-side is missing
        if 'avg_temp' in df.columns and 'temp_avg' in df.columns:
            df['avg_temp'] = df['avg_temp'].fillna(df['temp_avg'])

        # Clean up helper
        df = df.drop(columns=['ls_bin'], errors='ignore')
        return df

    # ============== Weather processing ==============

    def _create_weather_base(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        # Normalize datetime
        datetime_col = None
        for c in ['bin', 'datetime', 'timestamp']:
            if c in weather_df.columns:
                datetime_col = c
                break
        if datetime_col is None:
            raise ValueError("Weather data must contain a datetime column: one of ['bin','datetime','timestamp'].")

        w = weather_df.copy()
        w['datetime'] = pd.to_datetime(w[datetime_col], utc=True, errors='coerce')

        # Derive sol if absent
        if 'sol' not in w.columns:
            delta_days = (w['datetime'] - self.LANDING_EARTH_DT) / pd.Timedelta(days=1)
            w['sol'] = np.floor(delta_days / self.MARS_SOL_TO_EARTH_DAYS).astype('int64')
        w = w.dropna(subset=['sol']).copy()
        w['sol'] = w['sol'].astype('int32')

        # Ls normalization
        if 'solar_longitude_deg' in w.columns:
            w['solar_longitude_deg'] = pd.to_numeric(w['solar_longitude_deg'], errors='coerce')
        elif 'solar_longitude' in w.columns:
            w['solar_longitude_deg'] = pd.to_numeric(w['solar_longitude'], errors='coerce')
        else:
            w['solar_longitude_deg'] = np.nan

        # Sensor availability
        temp_cols = [c for c in ['BMY_BASE_ROD_TEMP', 'BMY_MID_ROD_TEMP', 'BMY_TIP_ROD_TEMP'] if c in w.columns]
        wind_speed_col = 'BMY_HORIZONTAL_WIND_SPEED' if 'BMY_HORIZONTAL_WIND_SPEED' in w.columns else None
        wind_dir_col = 'BMY_WIND_DIRECTION' if 'BMY_WIND_DIRECTION' in w.columns else None

        # Temperature aggregation
        if temp_cols:
            temp_long = w.melt(id_vars=['sol'], value_vars=temp_cols, var_name='temp_type', value_name='temp_val')
            temp_agg_base = temp_long.groupby('sol')['temp_val']
            temp_agg = temp_agg_base.agg(temp_min='min', temp_max='max', temp_avg='mean').reset_index()
            temp_agg['temp_range'] = temp_agg['temp_max'] - temp_agg['temp_min']
            # Quantiles for robustness
            tq = temp_agg_base.quantile([0.05, 0.5, 0.95]).unstack().reset_index().rename(
                columns={0.05: 'temp_p05', 0.5: 'temp_median', 0.95: 'temp_p95'})
            temp_agg = temp_agg.merge(tq, on='sol', how='left')
        else:
            temp_agg = pd.DataFrame({'sol': w['sol'].unique()})
            for c in ['temp_min', 'temp_max', 'temp_avg', 'temp_range', 'temp_p05', 'temp_median', 'temp_p95']:
                temp_agg[c] = np.nan

        # Wind speed aggregation
        if wind_speed_col:
            ws_group = w.groupby('sol')[wind_speed_col]
            ws_agg = ws_group.agg(wind_speed_avg='mean', wind_speed_max='max', wind_speed_std='std').reset_index()
            ws_med = ws_group.median().reset_index(name='wind_speed_median')
            ws_p95 = ws_group.quantile(0.95).reset_index(name='wind_speed_p95')
            ws_agg = ws_agg.merge(ws_med, on='sol', how='left').merge(ws_p95, on='sol', how='left')
            ws_agg['wind_gust_factor'] = ws_agg['wind_speed_max'] / (ws_agg['wind_speed_avg'].clip(lower=0.1))
        else:
            ws_agg = pd.DataFrame({'sol': w['sol'].unique()})
            for c in ['wind_speed_avg', 'wind_speed_max', 'wind_speed_std', 'wind_speed_median', 'wind_speed_p95', 'wind_gust_factor']:
                ws_agg[c] = np.nan

        # Circular wind direction aggregation (uniform weighting)
        if wind_dir_col:
            wd = w[['sol', wind_dir_col]].dropna()
            def circ_agg(g):
                a = np.deg2rad(g[wind_dir_col].values.astype(float))
                if a.size == 0:
                    return pd.Series({'wind_direction_avg': np.nan, 'wind_direction_std': np.nan, 'wind_resultant_length': np.nan})
                s, c = np.sin(a).mean(), np.cos(a).mean()
                ang = (np.degrees(np.arctan2(s, c)) + 360) % 360
                R = np.sqrt(s**2 + c**2)
                std_rad = np.sqrt(-2 * np.log(R)) if R > 0 else np.nan
                std_deg = np.degrees(std_rad) if pd.notna(std_rad) else np.nan
                return pd.Series({'wind_direction_avg': ang, 'wind_direction_std': std_deg, 'wind_resultant_length': R})
            wd_agg = wd.groupby('sol', as_index=False).apply(circ_agg).reset_index(drop=True)
        else:
            wd_agg = pd.DataFrame({'sol': w['sol'].unique(),
                                   'wind_direction_avg': np.nan,
                                   'wind_direction_std': np.nan,
                                   'wind_resultant_length': np.nan})

        # Speed-weighted wind vector mean (requires both speed and direction)
        if wind_speed_col and wind_dir_col:
            rad = np.deg2rad(w[wind_dir_col].astype(float))
            w['_u'] = w[wind_speed_col].astype(float) * np.cos(rad)
            w['_v'] = w[wind_speed_col].astype(float) * np.sin(rad)
            uv = w.groupby('sol')[['_u', '_v']].mean().reset_index().rename(columns={'_u': 'wind_u_mean', '_v': 'wind_v_mean'})
            uv['wind_vector_mean_speed'] = np.sqrt(uv['wind_u_mean']**2 + uv['wind_v_mean']**2)
            uv['wind_vector_mean_dir'] = (np.degrees(np.arctan2(uv['wind_v_mean'], uv['wind_u_mean'])) + 360) % 360
        else:
            uv = pd.DataFrame({'sol': w['sol'].unique(),
                               'wind_u_mean': np.nan, 'wind_v_mean': np.nan,
                               'wind_vector_mean_speed': np.nan, 'wind_vector_mean_dir': np.nan})

        # Ls aggregation
        ls_agg = w.groupby('sol', as_index=False).agg(solar_longitude_deg=('solar_longitude_deg', 'mean'))

        # Counts and completeness
        sensor_cols = temp_cols + ([wind_speed_col] if wind_speed_col else []) + ([wind_dir_col] if wind_dir_col else [])
        counts = w.groupby('sol', as_index=False).agg(weather_readings_count=('sol', 'size'))
        if temp_cols:
            def temp_compl(g):
                return g[temp_cols].notna().sum().sum() / (len(g) * len(temp_cols)) if len(g) > 0 else np.nan
            temp_compl_df = w.groupby('sol').apply(temp_compl).rename('weather_temp_completeness').reset_index()
        else:
            temp_compl_df = pd.DataFrame({'sol': w['sol'].unique(), 'weather_temp_completeness': np.nan})

        wind_cols = [c for c in [wind_speed_col, wind_dir_col] if c]
        if wind_cols:
            def wind_compl(g):
                return g[wind_cols].notna().sum().sum() / (len(g) * len(wind_cols)) if len(g) > 0 else np.nan
            wind_compl_df = w.groupby('sol').apply(wind_compl).rename('weather_wind_completeness').reset_index()
        else:
            wind_compl_df = pd.DataFrame({'sol': w['sol'].unique(), 'weather_wind_completeness': np.nan})

        if sensor_cols:
            def completeness(g):
                return g[sensor_cols].notna().sum().sum() / (len(g) * len(sensor_cols)) if len(g) > 0 else np.nan
            compl = w.groupby('sol').apply(completeness).rename('weather_data_completeness').reset_index()
        else:
            compl = pd.DataFrame({'sol': w['sol'].unique(), 'weather_data_completeness': np.nan})

        # Merge weather components
        base = (temp_agg.merge(ws_agg, on='sol', how='outer')
                        .merge(wd_agg, on='sol', how='outer')
                        .merge(uv, on='sol', how='outer')
                        .merge(ls_agg, on='sol', how='outer')
                        .merge(counts, on='sol', how='outer')
                        .merge(temp_compl_df, on='sol', how='outer')
                        .merge(wind_compl_df, on='sol', how='outer')
                        .merge(compl, on='sol', how='outer'))

        base['weather_available_today'] = (base['weather_readings_count'] > 0).astype('int8')

        # Fill numerics with medians where appropriate
        num_cols = [c for c in base.select_dtypes(include=[np.number]).columns if c != 'sol']
        for c in num_cols:
            if base[c].isna().any():
                base[c] = base[c].fillna(base[c].median())

        base = base.sort_values('sol').reset_index(drop=True)
        return base
    
    # ============== Activities processing ==============

    def _aggregate_activities(self, activities_df: pd.DataFrame) -> pd.DataFrame:
        df = activities_df.copy()

        # Normalize datetime/sol
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
        if 'sol' not in df.columns:
            raise ValueError("Activities data must contain 'sol'.")
        df['sol'] = pd.to_numeric(df['sol'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['sol']).copy()
        df['sol'] = df['sol'].astype('int32')

        # Ls normalization in activities
        if 'solar_longitude_deg' in df.columns:
            df['solar_longitude_deg'] = pd.to_numeric(df['solar_longitude_deg'], errors='coerce')
        elif 'solar_longitude' in df.columns:
            df['solar_longitude_deg'] = pd.to_numeric(df['solar_longitude'], errors='coerce')
        else:
            df['solar_longitude_deg'] = np.nan

        # Base aggregation
        agg = df.groupby('sol').agg(
            activity_count=('activity', 'size'),
            activity_text=('activity', lambda s: ' | '.join(s.dropna().astype(str))),
            solar_longitude_deg_act=('solar_longitude_deg', 'mean')
        ).reset_index()
        agg['has_activity'] = 1

        # Operational status rates if present (0=problem, 1=normal)
        if 'operational_status' in df.columns:
            ops = df.groupby('sol')['operational_status'].agg(['mean', 'min']).reset_index()
            ops = ops.rename(columns={'mean': 'operational_status_rate', 'min': 'ops_min'})
            agg = agg.merge(ops, on='sol', how='left')
            agg['has_operational_problem'] = (agg['ops_min'] == 0).fillna(0).astype('int8')
            agg = agg.drop(columns=['ops_min'])
        else:
            agg['operational_status_rate'] = np.nan
            agg['has_operational_problem'] = 0

        if 'criticality' in df.columns:
            crt = df.groupby('sol')['criticality'].agg(['mean', 'min']).reset_index()
            crt = crt.rename(columns={'mean': 'criticality_rate', 'min': 'crit_min'})
            agg = agg.merge(crt, on='sol', how='left')
            agg['is_critical_operation'] = (agg['crit_min'] == 0).fillna(0).astype('int8')
            agg = agg.drop(columns=['crit_min'])
        else:
            agg['criticality_rate'] = np.nan
            agg['is_critical_operation'] = 0

        if 'power_status' in df.columns:
            pwr = df.groupby('sol')['power_status'].agg(['mean', 'min']).reset_index()
            pwr = pwr.rename(columns={'mean': 'power_status_rate', 'min': 'pwr_min'})
            agg = agg.merge(pwr, on='sol', how='left')
            agg['has_power_issue'] = (agg['pwr_min'] == 0).fillna(0).astype('int8')
            agg = agg.drop(columns=['pwr_min'])
        else:
            agg['power_status_rate'] = np.nan
            agg['has_power_issue'] = 0

        agg['activity_count'] = agg['activity_count'].astype('int16')
        agg['activity_text'] = agg['activity_text'].fillna('')
        return agg

    # ============== Temporal and mission context ==============

    def _unify_and_fill_ls(self, df: pd.DataFrame) -> pd.Series:
        # Prefer weather Ls; fallback to activity Ls
        ls = df['solar_longitude_deg'].astype(float).copy()
        if 'solar_longitude_deg_act' in df.columns:
            ls = ls.fillna(df['solar_longitude_deg_act'].astype(float))

        # Interpolate Ls using angle unwrapping
        s = ls.astype(float)
        idx = s.index
        notna = s.notna()
        if not notna.any():
            return s
        rad = np.deg2rad(s[notna].values)
        unwrapped = np.unwrap(rad)
        unr = pd.Series(unwrapped, index=s[notna].index)
        interp = unr.reindex(idx).interpolate().ffill().bfill()
        deg = (np.degrees(interp) % 360.0).astype(float)
        return deg

    def _add_sol_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values('sol').reset_index(drop=True)
        df['mars_year'] = (df['sol'] // self.MARS_YEAR_SOLS + 1).astype('int16')
        df['sol_of_mars_year'] = (df['sol'] % self.MARS_YEAR_SOLS).astype('int16')

        # Sol gaps
        df['sol_gap'] = df['sol'].diff().fillna(1).astype('int16')
        df['is_consecutive_sol'] = (df['sol_gap'] == 1).astype('int8')
        df['sol_gap_size'] = np.clip(df['sol_gap'] - 1, 0, 30).astype('int16')

        # Normalized mission age (0-1)
        max_sol = max(df['sol'].max(), 1)
        df['mission_age_linear'] = (df['sol'] / max_sol).astype('float32')
        df['mission_age_log'] = (np.log1p(df['sol']) / np.log1p(max_sol)).astype('float32')
        return df

    def _add_mission_context(self, df: pd.DataFrame) -> pd.DataFrame:
        # InSight-tailored phase boundaries
        conditions = [
            df['sol'] <= 100,
            (df['sol'] > 100) & (df['sol'] <= 709),
            (df['sol'] > 709) & (df['sol'] <= 1060),
            df['sol'] > 1060
        ]
        phases = ['Checkout', 'Primary', 'Extended_1', 'Extended_2+']
        df['mission_phase'] = np.select(conditions, phases, default='Extended_2+')

        # Phase progress (0-1 within phase)
        phase_bounds = {
            'Checkout': (0, 100),
            'Primary': (101, 709),
            'Extended_1': (710, 1060),
            'Extended_2+': (1061, df['sol'].max())
        }
        df['phase_progress'] = 0.0
        for phase, (start, end) in phase_bounds.items():
            m = df['mission_phase'] == phase
            if m.any():
                prog = (df.loc[m, 'sol'] - start) / max(end - start, 1)
                df.loc[m, 'phase_progress'] = np.clip(prog, 0, 1)
        df['phase_progress'] = df['phase_progress'].astype('float32')

        # Aging factors (monotonic)
        df['equipment_age_factor'] = (0.3 * (1 - np.exp(-df['sol'] / 500.0))).astype('float32')
        df['wear_accumulation'] = np.minimum(df['sol'] / 1000.0, 1.0).astype('float32')
        return df

    # ============== Activity analysis ==============

    def _add_activity_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        df['activity_text'] = df.get('activity_text', '').fillna('').astype(str)
        text_lower = df['activity_text'].str.lower()

        # Counts
        df['activity_word_count'] = text_lower.str.split().str.len().fillna(0).astype('int16')
        df['instruction_complexity'] = (
            text_lower.str.count(',') +
            text_lower.str.count(';') +
            text_lower.str.count(r'\band\b')
        ).fillna(0).astype('int16') + 1

        # Multipart detection
        df['is_multipart_operation'] = text_lower.str.contains(r'part\s+\d+', regex=True, na=False).astype('int8')
        part_num = text_lower.str.extract(r'part\s+(\d+)', expand=False)
        df['operation_part_number'] = part_num.fillna(0).astype(int).astype('int16')

        # Instrument usage matrix
        for inst, kws in self.instrument_keywords.items():
            pattern = '|'.join([re.escape(k) for k in kws])
            df[f'uses_{inst.lower()}'] = text_lower.str.contains(pattern, regex=True, na=False).astype('int8')

        use_cols = [c for c in df.columns if c.startswith('uses_')]
        df['systems_involved'] = df[use_cols].sum(axis=1).astype('int16')

        # Primary instrument
        if use_cols:
            df['primary_instrument'] = np.where(df[use_cols].max(axis=1).gt(0),
                                                pd.DataFrame(df[use_cols].values, columns=use_cols).idxmax(axis=1)
                                                .str.replace('uses_', '').str.upper(),
                                                'NONE')
        else:
            df['primary_instrument'] = 'NONE'

        # Activity type matrix
        type_cols = []
        for atype, kws in self.activity_types.items():
            pattern = '|'.join([re.escape(k) for k in kws])
            col = f'is_{atype.lower()}_activity'
            df[col] = text_lower.str.contains(pattern, regex=True, na=False).astype('int8')
            type_cols.append(col)

        # Primary activity type
        if type_cols:
            df['primary_activity_type'] = np.where(df[type_cols].max(axis=1).gt(0),
                                                   pd.DataFrame(df[type_cols].values, columns=type_cols).idxmax(axis=1)
                                                   .str.replace('is_', '').str.replace('_activity', '').str.upper(),
                                                   'NONE')
        else:
            df['primary_activity_type'] = 'NONE'

        # FIXED: Robust normalization function
        def norm_by_q95(s):
            s = s.astype(float)
            if not s.notna().any():
                return pd.Series(0.0, index=s.index, dtype='float32')
            q = s.quantile(0.95)
            m = s.max(skipna=True)
            denom = q if pd.notna(q) and q > 0 else (m if pd.notna(m) and m > 0 else 1.0)
            out = (s / denom).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1).astype('float32')
            return out

        wc_n = norm_by_q95(df['activity_word_count'].astype(float))
        ic_n = norm_by_q95(df['instruction_complexity'].astype(float))
        sys_n = norm_by_q95(df['systems_involved'].astype(float))
        df['activity_intensity'] = (0.3 * wc_n + 0.4 * ic_n + 0.3 * sys_n).astype('float32')

        # Convenience
        df['multiple_activities'] = (df.get('activity_count', 0).fillna(0).astype(int) > 1).astype('int8')
        return df

    # ============== Environmental features ==============

    def _add_environmental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ls = df['solar_longitude_deg'].astype(float)

        # Seasons by Ls
        season_conditions = [
            (ls >= 0) & (ls < 90),
            (ls >= 90) & (ls < 180),
            (ls >= 180) & (ls < 270),
            (ls >= 270) & (ls < 360)
        ]
        seasons = ['Spring_North', 'Summer_North', 'Fall_North', 'Winter_North']
        df['mars_season'] = np.select(season_conditions, seasons, default='Spring_North')

        # Dust seasonal risk (triangular peak at Ls=240 within 180–300)
        risk = np.zeros(len(df), dtype=float)
        in_season = (ls >= self.DUST_SEASON_LS[0]) & (ls <= self.DUST_SEASON_LS[1])
        risk[in_season] = 1.0 - np.abs(ls[in_season] - 240.0) / 60.0
        df['dust_season_risk'] = np.clip(risk, 0.0, 1.0).astype('float32')

        # Orbital thermal (insolation proxy)
        df['orbital_thermal_factor'] = (np.cos(np.radians(ls - self.PERIHELION_LS)) * 0.5 + 0.5).astype('float32')

        # Ls cyclic encodings and daily change
        df['ls_sin'] = np.sin(np.radians(ls)).astype('float32')
        df['ls_cos'] = np.cos(np.radians(ls)).astype('float32')
        # Ls diff (unwrap to avoid 360° jump)
        rad = np.radians(ls.to_numpy())
        unwrapped = np.unwrap(rad)
        dls = np.degrees(np.diff(unwrapped, prepend=unwrapped[0]))
        df['solar_longitude_diff'] = dls.astype('float32')

        # Temperature volatility and trend
        if 'temp_avg' in df.columns:
            ta = df['temp_avg'].astype(float)
            df['temp_volatility_7sol'] = ta.rolling(7, min_periods=2).std().fillna(0).astype('float32')
            df['temp_trend_7sol'] = ((ta - ta.shift(6)) / 6.0).fillna(0).astype('float32')
        else:
            df['temp_volatility_7sol'] = 0.0
            df['temp_trend_7sol'] = 0.0

        return df

    # ============== Operational features ==============

    def _add_operational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values('sol').reset_index(drop=True)

        # If not aggregated from inputs, attempt text-based detection
        if 'has_operational_problem' not in df.columns:
            text = df['activity_text'].str.lower().fillna('')
            problem_keywords = ['safe mode', 'fault', 'error', 'fail', 'recovery', 'unsafe', 'safed',
                                'troubleshoot', 'diagnostic', 'anomaly', 'issue']
            df['has_operational_problem'] = text.str.contains('|'.join(problem_keywords), regex=True, na=False).astype('int8')

        if 'is_critical_operation' not in df.columns:
            text = df['activity_text'].str.lower().fillna('')
            critical_keywords = ['critical', 'priority', 'urgent', 'essential']
            df['is_critical_operation'] = text.str.contains('|'.join(critical_keywords), regex=True, na=False).astype('int8')

        if 'has_power_issue' not in df.columns:
            text = df['activity_text'].str.lower().fillna('')
            power_keywords = ['power issue', 'brownout', 'low power', 'heater fault']
            df['has_power_issue'] = text.str.contains('|'.join(power_keywords), regex=True, na=False).astype('int8')

        # Rolling operational tempo (3,7,15 sols)
        df['operational_tempo_3sol'] = df['activity_intensity'].rolling(3, min_periods=1).mean().astype('float32')
        df['operational_tempo_7sol'] = df['activity_intensity'].rolling(7, min_periods=1).mean().astype('float32')
        df['operational_tempo_15sol'] = df['activity_intensity'].rolling(15, min_periods=1).mean().astype('float32')

        # Since/To next problem (vectorized)
        mask = df['has_operational_problem'].astype(bool)
        last_evt = df['sol'].where(mask).ffill()
        next_evt = df['sol'].where(mask).bfill()
        df['sols_since_last_problem'] = (df['sol'] - last_evt).fillna(999).astype('int16')
        df['sols_to_next_problem'] = (next_evt - df['sol']).fillna(999).astype('int16')

        # Activity recency
        has_act = df.get('has_activity', pd.Series(0, index=df.index)).astype(bool)
        last_act = df['sol'].where(has_act).ffill()
        df['sols_since_last_activity'] = (df['sol'] - last_act).fillna(999).astype('int16')

        # Activity frequency
        df['activity_days_30sol'] = has_act.astype('int8').rolling(30, min_periods=1).sum().astype('int16')
        df['cumulative_activity_count'] = df.get('activity_count', pd.Series(0, index=df.index)).fillna(0).astype(int).cumsum().astype('int32')

        return df

    # ============== Dust storm features ==============

    def _add_dust_storm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'wind_speed_max' in df.columns and 'wind_speed_avg' in df.columns:
            ws_max = df['wind_speed_max'].astype(float)
            ws_avg = df['wind_speed_avg'].astype(float)

            # Gustiness and volatility
            df['wind_gustiness'] = (ws_max - ws_avg).clip(lower=0).astype('float32')
            df['wind_volatility_7sol'] = ws_avg.rolling(7, min_periods=2).std().fillna(0).astype('float32')

            # Dynamic thresholds (q90)
            q90_max = ws_max.quantile(0.90)
            if pd.isna(q90_max) or q90_max <= 0:
                q90_max = ws_max.max() if ws_max.max() > 0 else 1.0
            q90_gust = df['wind_gustiness'].quantile(0.90)
            if pd.isna(q90_gust) or q90_gust <= 0:
                q90_gust = df['wind_gustiness'].max() if df['wind_gustiness'].max() > 0 else 1.0
            
            # FIXED: Robust volatility scaling
            denom_vol = df['wind_volatility_7sol'].quantile(0.90)
            if not (pd.notna(denom_vol) and denom_vol > 0):
                denom_vol = 1.0
            vol_scaled = (df['wind_volatility_7sol'] / denom_vol).fillna(0).clip(0, 1)

            df['wind_dust_lifting'] = (ws_max > q90_max).astype('int8')
            df['wind_sustained_high'] = (df['wind_dust_lifting'].rolling(3, min_periods=1).sum() >= 2).astype('int8')
            df['wind_high_volatility'] = (df['wind_volatility_7sol'] > denom_vol).astype('int8')

            gustiness_scaled = (df['wind_gustiness'] / q90_gust).clip(0, 1)

            # Temperature cooling signal (if available)
            if 'temp_trend_7sol' in df.columns:
                t10 = df['temp_trend_7sol'].quantile(0.10)
                temp_drop = (df['temp_trend_7sol'] < t10).astype('int8')
            else:
                temp_drop = pd.Series(0, index=df.index)

            wind_component = 0.5 * df['wind_dust_lifting'] + 0.3 * gustiness_scaled + 0.2 * vol_scaled
            # FIXED: Ensure wind_component is NaN-safe
            if isinstance(wind_component, pd.Series):
                wind_component = wind_component.fillna(0)
            
            df['dust_storm_probability'] = np.clip(
                0.5 * wind_component + 0.4 * df['dust_season_risk'] + 0.1 * temp_drop, 0, 1
            ).astype('float32')
        else:
            df['wind_gustiness'] = 0.0
            df['wind_volatility_7sol'] = 0.0
            df['wind_dust_lifting'] = 0
            df['wind_sustained_high'] = 0
            df['wind_high_volatility'] = 0
            df['dust_storm_probability'] = df['dust_season_risk'].astype('float32')

        # Event level classification
        df['dust_event_level'] = 'none'
        df.loc[df['dust_storm_probability'] > 0.3, 'dust_event_level'] = 'minor'
        df.loc[df['dust_storm_probability'] > 0.5, 'dust_event_level'] = 'moderate'
        df.loc[df['dust_storm_probability'] > 0.7, 'dust_event_level'] = 'major'

        return df

    # ============== Equipment features ==============

    def _add_equipment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Cumulative usage per instrument and intensity
        use_cols = [c for c in df.columns if c.startswith('uses_')]
        for c in use_cols:
            base = c.replace('uses_', '')
            df[f'{base}_cumulative_usage'] = df[c].astype('int16').cumsum().astype('int16')
            recent = df[c].rolling(30, min_periods=1).sum().astype('float32')
            total = df[f'{base}_cumulative_usage'].replace(0, np.nan)
            df[f'{base}_usage_intensity'] = (recent / total).fillna(0).astype('float32')

        # Ensure has_activity exists
        df['has_activity'] = df.get('has_activity', 0).fillna(0).astype('int8')
        df['mission_operational_cycles'] = df['has_activity'].cumsum().astype('int16')

        # Reliability score (decreases with age and usage)
        age_deg = 0.2 * (1 - np.exp(-df['sol'] / 800.0))
        usage_deg = np.minimum(df['mission_operational_cycles'] / 1000.0, 0.3)
        df['equipment_reliability_score'] = np.clip(1.0 - age_deg - usage_deg, 0.3, 1.0).astype('float32')

        # Maintenance detection from text
        text = df['activity_text'].str.lower().fillna('')
        maint_keywords = ['maintenance', 'calibration', 'checkout', 'characterization']
        df['maintenance_activity'] = text.str.contains('|'.join(maint_keywords), regex=True, na=False).astype('int8')

        # Since last maintenance
        m = df['maintenance_activity'].astype(bool)
        last_m = df['sol'].where(m).ffill()
        df['sols_since_maintenance'] = (df['sol'] - last_m).fillna(999).astype('int16')

        return df

    # ============== Compound variables (powerful indices) ==============

    def _add_compound_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        # FIXED: Robust scaling function for compound variables
        def scale_q95(s):
            s = s.astype(float)
            if not s.notna().any():
                return pd.Series(0.0, index=s.index, dtype='float32')
            q = s.quantile(0.95)
            m = s.max(skipna=True)
            denom = q if pd.notna(q) and q > 0 else (m if pd.notna(m) and m > 0 else 1.0)
            return (s / denom).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1).astype('float32')

        # Environmental stress index: wind volatility, temp volatility, dust seasonal risk
        wind_vol_n = scale_q95(df.get('wind_volatility_7sol', pd.Series(0, index=df.index)).astype(float))
        temp_vol_n = scale_q95(df.get('temp_volatility_7sol', pd.Series(0, index=df.index)).astype(float))
        dust_risk = df['dust_season_risk'].astype(float)

        df['environmental_stress_index'] = (0.4 * wind_vol_n + 0.3 * temp_vol_n + 0.3 * dust_risk).astype('float32')

        # FIXED: Operational risk index with NaN-safe handling
        dust_p = df['dust_storm_probability'].fillna(df['dust_season_risk']).astype('float32')
        inv_rel = (1.0 - df['equipment_reliability_score']).fillna(0).clip(0, 1).astype('float32')
        tempo = df.get('operational_tempo_7sol', pd.Series(0, index=df.index)).fillna(0).clip(0, 1).astype('float32')
        df['operational_risk_index'] = (0.4 * dust_p + 0.35 * inv_rel + 0.25 * tempo).clip(0, 1).astype('float32')

        # Power availability proxy: orbital insolation reduced by seasonal dust risk
        df['power_availability_proxy'] = (df['orbital_thermal_factor'] * (1.0 - df['dust_season_risk'])).astype('float32')

        return df

    # ============== Optimization ==============

    def _optimize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values('sol').reset_index(drop=True)

        # Remove obvious raw ID/time columns if present (sol-level focus)
        drop_cols = [c for c in ['year', 'month', 'day', 'hour', 'day_of_year', 'weekday',
                                 'date', 'datetime', 'end', 'look_ahead', 'file_id', 'file_id_str',
                                 'solar_longitude_deg_act'] if c in df.columns]
        df = df.drop(columns=drop_cols, errors='ignore')

        # Keep both Ls columns for compatibility and clarity
        if 'solar_longitude_deg' in df.columns:
            df['solar_longitude'] = df.get('solar_longitude', df['solar_longitude_deg'])

        # Downcast numerics to reduce memory without dropping columns
        for c in df.select_dtypes(include=['float64']).columns:
            df[c] = pd.to_numeric(df[c], downcast='float')
        for c in df.select_dtypes(include=['int64']).columns:
            if c != 'sol':
                df[c] = pd.to_numeric(df[c], downcast='integer')

        return df


def create_mars_master_dataset(activities_file='mars_activities_dataframe.csv',
                               weather_file='mars_weather_15min_with_solar_longitude.csv',
                               output_csv='mars_mission_master_dataset_optimized.csv'):
    creator = MarsMasterDatasetCreator()
    master_df = creator.create_master_dataset(activities_file=activities_file, weather_file=weather_file)
    master_df.to_csv(output_csv, index=False)
    return master_df


if __name__ == '__main__':
    df = create_mars_master_dataset(
        activities_file='mars_activities_dataframe.csv',
        weather_file='mars_weather_15min_with_solar_longitude.csv',
        output_csv='mars_mission_master_dataset_optimized.csv'
    )
    print(df.shape)
    print(df.head())