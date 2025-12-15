def get_SPFs(df):

    HP = df['Heat_Pump_Energy_Output'].iloc[-1] - df['Heat_Pump_Energy_Output'].iloc[0]
    WS = df['Whole_System_Energy_Consumed'].iloc[-1] - df['Whole_System_Energy_Consumed'].iloc[0]
    CP = df['Circulation_Pump_Energy_Consumed'].iloc[-1] - df['Circulation_Pump_Energy_Consumed'].iloc[0]
    
    IH = (df['Immersion_Heater_Energy_Consumed'].iloc[-1] - df['Immersion_Heater_Energy_Consumed'].iloc[0]) if 'Immersion_Heater_Energy_Consumed' in df.columns else 0
    BU = (df['Back-up_Heater_Energy_Consumed'].iloc[-1] - df['Back-up_Heater_Energy_Consumed'].iloc[0]) if 'Back-up_Heater_Energy_Consumed' in df.columns else 0
    
    # SPF formulas (now unified for all cases)
    SPF_H2 = HP / (WS - CP - IH - BU)
    SPF_H3 = (HP + IH + BU) / (WS - CP)
    SPF_H4 = (HP + IH + BU) / WS

    return SPF_H2, SPF_H3, SPF_H4


def create_heat_event_dict(df, HP_cap_2min):
    """
    Identify and analyze space heating events from a dataframe of time-indexed data.

    Assumes df has at least:
    - 'Elec_in'   : electrical input per timestep
    - 'Heat_out'  : heat output per timestep
    - 'External_Air_Temperature'
    - 'Internal_Air_Temperature'
    - 'Heat_Pump_Heating_Flow_Temperature'
    - 'Heat_Pump_Return_Temperature'
    """

    df = df.copy()
    # 1) If Heat_out is NaN -> set Heat_out = 0 and Elec_in = 0
    df.loc[df['Heat_out'].isna(), ['Heat_out', 'Elec_in']] = 0
    # 2) If Elec_in < 0.01 -> set Elec_in = 0
    df.loc[df['Elec_in'] < 0.01, 'Elec_in'] = 0

    # 3) Cap Heat_out at HP_cap_2min *per timestep capacity
    df['Heat_out'] = df['Heat_out'].clip(upper=HP_cap_2min)

    # --- identify heating events based on Elec_in > 0 ---
    heat_ind = np.where(df['Elec_in'] > 0)[0]
    heat_event_inds = []
    start_ind, end_ind = 0, 0

    for ind in heat_ind:
        if ind > end_ind:
            # find end of this contiguous heating period
            for j in range(ind, len(df)):
                if j not in heat_ind:
                    start_ind = ind
                    end_ind = j
                    break
            heat_event_inds.append([start_ind, end_ind])

    # --- build event dictionary ---
    heat_events = {}
    for j, (start, end) in enumerate(heat_event_inds):
        idx_slice = df.index[start:end]

        electricity = df.loc[idx_slice, 'Elec_in'].sum()
        heat        = df.loc[idx_slice, 'Heat_out'].sum()
        av_temp     = df.loc[idx_slice, 'External_Air_Temperature'].mean()
        av_int      = df.loc[idx_slice, 'Internal_Air_Temperature'].mean()
        av_flow     = df.loc[idx_slice, 'Heat_Pump_Heating_Flow_Temperature'].mean()
        av_return   = df.loc[idx_slice, 'Heat_Pump_Return_Temperature'].mean()

        heat_events[j] = {
            'start_index': start,
            'end_index': end,
            'electricity': electricity,
            'heat': heat,
            'av. temp': av_temp,
            'av. int.': av_int,
            'av. flow': av_flow,
            'av. return': av_return,
        }

        heat_events[j]['delta T1'] = av_return - av_temp
        heat_events[j]['COP'] = heat / electricity if electricity != 0 else np.nan
        heat_events[j]['LF'] = heat / ((end - start) * HP_cap_2min)
        heat_events[j]['mid time'] = df.index[start] + (df.index[end] - df.index[start]) / 2

    # --- remove bad events ---
    problems = []
    for h, event in heat_events.items():
        if (
            np.isnan(event['electricity']) or
            np.isnan(event['heat']) or
            np.isnan(event['COP']) or
            np.isnan(event['av. temp']) or
            np.isnan(event['av. return']) or
            np.isnan(event['av. int.']) or
            (event['av. flow'] < event['av. return'])
        ):
            problems.append(h)

    for p in problems:
        heat_events.pop(p)

    return heat_events


