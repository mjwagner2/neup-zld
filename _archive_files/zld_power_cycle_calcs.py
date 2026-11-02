# this file provides preliminary calculations for the zld + tes model


def compute_power_cycle_nominals(
        
    Q_dot               = 9.5e5,   # units: kWth,     thermal output from reactor
    eta                 = 0.52,    # units: --,       nominal cycle efficiency
    Cp_salt             = 1.5,     # units: kJ/kg-K,  specific heat of salt
    T_salt_hot          = 565,     # units: degC,     salt hot temperature
    T_salt_cold         = 330,     # units: degC,     salt cold temperature
    Delta_H_hx1_steam   = 2126,    # units: kJ/kg,    steam enthalpy change across hx1
    Delta_H_hx2_steam   = 2494,    # units: kJ/kg,    steam enthalpy change across hx2
    Delta_H_hx2_water   = 421.3,   # units: kJ/kg,    water enthalpy change across hx2
    F_ext               = 0.53     # units: --,       nominal steam extraction 
):
    
    
    """
    compute nominal power cycle values and return a dictionary 
    to use in pyomo model
    """


    W_dot_ref           = eta*Q_dot                                                  # units: kWe,        reference power output from cycle
    
    
    # hx1 
    Delta_T_hx1_salt    = T_salt_hot-T_salt_cold                                     # units: degC,       temperature difference in salt storage
    M_dot_hx1_salt_nom  = Q_dot/(Cp_salt*Delta_T_hx1_salt)                           # units: kg/s,       nominal salt mass flow
    M_dot_hx1_steam_nom = Q_dot/Delta_H_hx1_steam                                    # units: kg/s,       nominal steam mass flow 
    K_hx1               = M_dot_hx1_steam_nom/M_dot_hx1_salt_nom                     # units: --,         steam mass flow per unit salt mass flow in hx1


    # hx2
    M_dot_hx2_steam_nom = M_dot_hx1_steam_nom*F_ext                                  # units: kg/s,       nominal steam extraction mass flow
    M_dot_hx2_water_nom = M_dot_hx2_steam_nom*Delta_H_hx2_steam/Delta_H_hx2_water    # units: kg/s,       nominal water mass flow
    K_hx2               = M_dot_hx2_water_nom/M_dot_hx2_steam_nom                    # units: --,         water mass flow per unit steam mass flow in hx2 


    # power conversions
    K_power             = W_dot_ref/M_dot_hx1_steam_nom                              # units: kWe/kg/s,   electricity generated per unit mass flow of steam
    K_loss              = W_dot_ref/M_dot_hx2_steam_nom                              # units: kWe/kg/s,   electricity loss per unit mass flow of steam extracted
        
     
    
    return {
        "Q_dot":               Q_dot,
        "eta":                 eta,
        "W_dot_ref":           W_dot_ref,
        "Cp_salt":             Cp_salt,
        "Delta_T_hx1_salt":    Delta_T_hx1_salt,
        "Delta_H_hx2_water":   Delta_H_hx2_water,
        "M_dot_hx1_salt_nom":  M_dot_hx1_salt_nom,
        "M_dot_hx1_steam_nom": M_dot_hx1_steam_nom,
        "K_hx1":               K_hx1,
        "M_dot_hx2_steam_nom": M_dot_hx2_steam_nom,
        "M_dot_hx2_water_nom": M_dot_hx2_water_nom,
        "K_hx2":               K_hx2,
        "K_power":             K_power,
        "K_loss":              K_loss,
        "F_ext":               F_ext 
    }






























