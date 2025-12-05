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

