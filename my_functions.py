import numpy as np
from scipy.optimize import curve_fit
from sklearn import linear_model
import pandas as pd

def get_SPFs(df):

    HP = df['Heat_Pump_Energy_Output'].iloc[-1] - df['Heat_Pump_Energy_Output'].iloc[0]
    WS = df['Whole_System_Energy_Consumed'].iloc[-1] - df['Whole_System_Energy_Consumed'].iloc[0]
    CP = df['Circulation_Pump_Energy_Consumed'].iloc[-1] - df['Circulation_Pump_Energy_Consumed'].iloc[0]
    
    IH = (df['Immersion_Heater_Energy_Consumed'].iloc[-1] - df['Immersion_Heater_Energy_Consumed'].iloc[0]) if 'Immersion_Heater_Energy_Consumed' in df.columns else 0
    BU = (df['Back-up_Heater_Energy_Consumed'].iloc[-1] - df['Back-up_Heater_Energy_Consumed'].iloc[0]) if 'Back-up_Heater_Energy_Consumed' in df.columns else 0
    
    # Calculate SPFs
    SPF_H2 = HP / (WS - CP - IH - BU)
    SPF_H3 = (HP + IH + BU) / (WS - CP)
    SPF_H4 = (HP + IH + BU) / WS

    return SPF_H2, SPF_H3, SPF_H4


def create_heat_event_dict(df, HP_cap_2min):
    """
    Assumes df has at least:
    - 'Elec_in'   : electrical input per timestep
    - 'Heat_out'  : heat output per timestep
    - 'External_Air_Temperature'
    - 'Internal_Air_Temperature'
    - 'Heat_Pump_Heating_Flow_Temperature'
    - 'Hot_Water_Flow_Temperature'
    - 'Heat_Pump_Return_Temperature'
    """

    df = df.copy()

    # --- infer mode per timestep ---
    sh = df['Heat_Pump_Heating_Flow_Temperature'].notna()
    hw = df['Hot_Water_Flow_Temperature'].notna()
    # there should be no overlap between heat and hot water, but sometimes there is, so tidy up
    overlap = sh & hw
    if overlap.any():
        # print(f"Resolving {int(overlap.sum())} overlap timestep(s) → defaulting to hot water")
        df.loc[overlap, 'Heat_Pump_Heating_Flow_Temperature'] = np.nan
    
    # masks after correction
    sh = df['Heat_Pump_Heating_Flow_Temperature'].notna()
    hw = df['Hot_Water_Flow_Temperature'].notna()
    # create a new array to mark when the events are space and hot water
    mode = np.full(len(df), None, dtype=object)
    mode[sh.to_numpy()] = 'space_heat'
    mode[hw.to_numpy()] = 'hot_water'
    df['mode'] = mode

    # --- heating running mask ---
    heat_on = (df['Elec_in'].to_numpy() > 0)

    # --- create heat event indices
    heat_event_inds = []
    i, n = 0, len(df)
    while i < n:
        if not heat_on[i] or df['mode'].iloc[i] is None:
            i += 1
            continue

        this_mode = df['mode'].iloc[i]
        start = i
        i += 1

        # for each heating event, increment to find the end of that event
        # note if the event changes type that is the end
        while i < n and heat_on[i] and (df['mode'].iloc[i] == this_mode):
            i += 1

        end = i  # exclusive
        heat_event_inds.append((start, end, this_mode))

    # --- build event dictionary ---
    heat_events = {}
    for j, (start, end, event_type) in enumerate(heat_event_inds):
        idx_slice = df.index[start:end]
        # get the average over the event
        electricity = df.loc[idx_slice, 'Elec_in'].sum()
        heat        = df.loc[idx_slice, 'Heat_out'].sum()
        av_temp     = df.loc[idx_slice, 'External_Air_Temperature'].mean()
        av_int      = df.loc[idx_slice, 'Internal_Air_Temperature'].mean()
        av_return   = df.loc[idx_slice, 'Heat_Pump_Return_Temperature'].mean()

        if event_type == 'space_heat':
            av_flow = df.loc[idx_slice, 'Heat_Pump_Heating_Flow_Temperature'].mean()
        else:  # 'hot_water'
            av_flow = df.loc[idx_slice, 'Hot_Water_Flow_Temperature'].mean()

        heat_events[j] = {
            'start_index': start,
            'end_index': end,
            'event_type': event_type,
            'electricity': electricity,
            'heat': heat,
            'av temp': av_temp,
            'av int': av_int,
            'av flow': av_flow,
            'av return': av_return,
        }

        heat_events[j]['deltaT'] = av_return - av_temp
        heat_events[j]['COP'] = heat / electricity if electricity != 0 else np.nan
        heat_events[j]['LF'] = heat / ((end - start) * HP_cap_2min) if (end - start) > 0 else np.nan
        heat_events[j]['mid time'] = df.index[start] + (df.index[end - 1] - df.index[start]) / 2

    # --- remove bad events ---
    problems = []
    for h, event in heat_events.items():
        if (
            np.isnan(event['electricity']) or
            np.isnan(event['heat']) or
            np.isnan(event['COP']) or
            np.isnan(event['av temp']) or
            np.isnan(event['av return']) or
            np.isnan(event['av flow']) or # remove events with missing data
            (event['av flow'] < event['av return']) or # remove events with nonsensical flow
            (event['deltaT'] <= 0) # remove events which have outdoor temp >= indoor temp
        ):
            problems.append(h)

    for p in problems:
        heat_events.pop(p)

    key_list = heat_events.keys()
    
    # --- total delivered heat by mode (from raw dataframe) ---
    total_sh = df.loc[sh, 'Heat_out'].sum()
    total_hw = df.loc[hw, 'Heat_out'].sum()
    
    # --- heat accounted for by detected events ---
    if total_sh > 0:
        sh_accounted_for = np.sum([heat_events[j]['heat'] for j in key_list if heat_events[j]['event_type'] == 'space_heat']) / total_sh
    else: 
        sh_accounted_for = np.nan
    if total_hw > 0: 
        hw_accounted_for = np.sum([heat_events[j]['heat'] for j in key_list if heat_events[j]['event_type'] == 'hot_water']) / total_hw
    else:
        hw_accounted_for = np.nan
        
    # print("\n--- Heat accounting summary ---")
    # print(f"Total space heat delivered: {total_sh:.2f}")
    # print(f"Space heat in events:       {total_sh * sh_accounted_for:.2f}")
    # print(f"Space heat represented:     {100 * sh_accounted_for:.2f}%")111
    # print()
    # print(f"Total hot water delivered:  {total_hw:.2f}")
    # print(f"Hot water in events:        {total_hw * hw_accounted_for:.2f}")
    # print(f"Hot water represented:      {100 * hw_accounted_for:.2f}%")

    return heat_events, sh_accounted_for, hw_accounted_for


def fill_missing_data(df, column, degree=2):
    series = df[column].copy()
    is_nan = series.isna()

    # Identify contiguous gaps
    groups = (is_nan != is_nan.shift()).cumsum()
    for grp in series[is_nan].groupby(groups):
        idxs = grp[1].index
        gap_len = len(idxs)

        if gap_len < 24:
            gap_start = idxs[0]
            gap_end = idxs[-1]
            after_gap = gap_end + pd.Timedelta(hours=1)

            # Define fitting windows
            before_start = gap_start - pd.Timedelta(hours=12)
            after_end = after_gap + pd.Timedelta(hours=11)

            # Extract fitting data
            window_data = series.loc[before_start:gap_start - pd.Timedelta(hours=1)].dropna()
            window_data = pd.concat([window_data, series.loc[after_gap:after_end].dropna()])

            if len(window_data) < degree + 1:
                continue  # not enough data to fit

            # Use hours since window start as x values
            x_vals = (window_data.index - window_data.index[0]).total_seconds() / 3600
            y_vals = window_data.values

            coefs = np.polyfit(x_vals, y_vals, degree)
            poly = np.poly1d(coefs)

            # Predict values for the entire gap
            for ts in idxs:
                x_pred = (ts - window_data.index[0]).total_seconds() / 3600
                series.loc[ts] = poly(x_pred)

        else:
            # Fill with previous day's data
            for ts in idxs:
                prev_day = ts - pd.Timedelta(days=1)
                if prev_day in series.index:
                    series.loc[ts] = series.loc[prev_day]

    df[column] = series
    return df

def fit_cop_models(
    temperatures,
    COPs,
    heat_values,
    event_types,
    a_thresh=10,
    b_thresh=1.5,
    cop_max=5,
    heat_min=1,
    do_plot=False,
    ax=None,
    scale=20,
    x_min=10,
    x_max=50,
    n_fit_points=100,
):
    """
    Filters COP data, fits inverse + linear COP models, selects best by R2

    Returns a dict with:
      - best_model_type ('inv' or 'lin')
      - best_params
      - R2_inv, R2_lin, R2_best
    """

    # Define the two fit functions (as requested)
    def model_inverse(x, a, b):
        return a / x - b

    def model_linear(x, a, b):
        return a * x - b

    def r_squared(true, predicted):
        TSS = np.sum((true - np.mean(true))**2)
        SSE = np.sum((true - predicted)**2)
        return 1 - SSE / TSS

    # Threshold curve
    min_COPs = a_thresh / temperatures + b_thresh

    # Outlier mask
    out_marker = (COPs > min_COPs) & (COPs < cop_max) & (heat_values > heat_min)

    temp_exc_outliers = temperatures[out_marker]
    COP_exc_outliers = COPs[out_marker]
    heat_exc_outliers = heat_values[out_marker]
    event_exc_outliers = np.array(event_types)[out_marker]

    # Map event types to colours
    color_map = {
        "space_heat": "tab:blue",
        "hot_water": "tab:orange",
    }

    # Fit both models
    params_inv, _ = curve_fit(model_inverse, temp_exc_outliers, COP_exc_outliers)
    params_lin, _ = curve_fit(model_linear, temp_exc_outliers, COP_exc_outliers)

    # Predictions + R2
    pred_inv = model_inverse(temp_exc_outliers, *params_inv)
    pred_lin = model_linear(temp_exc_outliers, *params_lin)

    R2_inv = r_squared(COP_exc_outliers, pred_inv)
    R2_lin = r_squared(COP_exc_outliers, pred_lin)

    # Select best
    if R2_inv > R2_lin:
        best_model = model_inverse
        best_params = params_inv
        best_type = 'inv'
        R2_best = R2_inv
    else:
        best_model = model_linear
        best_params = params_lin
        best_type = 'lin'
        R2_best = R2_lin

    # Optional plot
    if do_plot:
        facecolors = [color_map[e] for e in event_exc_outliers]
        ax.scatter(
            temp_exc_outliers,
            COP_exc_outliers,
            alpha=0.2,
            s=scale * heat_exc_outliers,
            facecolors=facecolors,
            edgecolors='k',
        )

        ax.set_ylim([1, 6])
        ax.set_yticks(np.arange(1, 6))
        ax.set_yticks(np.arange(1, 6, 0.2), minor=True)

        ax.set_xlim([9, 41])
        ax.set_xticks(np.arange(10, 51, 10))
        ax.set_xticks(np.arange(10, 51, 2), minor=True)

        # Legend markers for bubble sizes
        msizes = [1, 4, 10]
        for size in msizes:
            ax.scatter([], [], s=scale * size, label=f"{size} kWh",
                       facecolors='b', edgecolors='k', alpha=0.2)

        # Plot best fit line
        xd = np.linspace(x_min, x_max, n_fit_points)
        ffit = best_model(xd, *best_params)
        ax.plot(xd, ffit, linewidth=2, linestyle='--', color='k', label='fit')

        ax.text(0.5, 0.85, f"R²={R2_best:.2f}", transform=ax.transAxes, fontsize=10)
        ax.legend(loc='best', fontsize=8)

    return {
        'out_marker': out_marker,
        'temp_exc_outliers': temp_exc_outliers,
        'COP_exc_outliers': COP_exc_outliers,
        'best_type': best_type,
        'best_params': best_params,
        'R2_inv': float(R2_inv),
        'R2_lin': float(R2_lin),
        'R2_best': float(R2_best),
    }
