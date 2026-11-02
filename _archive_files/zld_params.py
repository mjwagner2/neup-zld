import pyomo.environ as pyo

from core.zld_power_cycle_calcs import compute_power_cycle_nominals

  
    
def add_sets(model):
   
    # processes 
    model.processes = pyo.Set(initialize=['RO', 'MED', 'ED', 'CRY'])


    # ions
    model.ions      = pyo.Set(initialize=['Li', 'Na', 'Cl'])


    # links between processes
    model.links     = pyo.Set(dimen=2,initialize=[
            ('SW', 'RO'),
            ('SW', 'MED'),
            ('SW', 'ED'),
            ('SW', 'CRY'),
            ('RO', 'MED'),
            ('RO', 'ED'),
            ('RO', 'CRY'),
            ('MED', 'ED'),
            ('MED', 'CRY'),
            ('ED', 'CRY')])
  
    
  
  
    
def add_time_set(model, N):
    
    model.nt      = pyo.Param(initialize=N)                 # units: --,    number of timesteps
    model.T       = pyo.Set(initialize=range(1,N+1))        # units: --,    set of timesteps
    model.Delta_t = pyo.Param(initialize=1)                 # units: hr,    timestep increments
   
    
   
   
   
def add_time_series_params(model, sched_price_elec, sched_cyc_eff):
    

    # electricity pricing schedule (units: $/kWh)
    model.Price_elec = pyo.Param(model.T,initialize={t:float(sched_price_elec[t-1]/1000.0) for t in model.T})


    # cycle efficiency schedule (units: --)
    model.Eff_cycle  = pyo.Param(model.T, initialize={t:float(sched_cyc_eff[t-1]) for t in model.T})


   

    
def add_thermo_params(model, oversize_factor=1.0):
    
    thermo_ref = compute_power_cycle_nominals()
    
    # nominal power cycle size
    model.W_dot_ref           = pyo.Param(initialize=thermo_ref["W_dot_ref"])
    
    
    # nominal mass flows
    model.M_dot_hx1_steam_nom = pyo.Param(initialize=thermo_ref["M_dot_hx1_steam_nom"])
    model.M_dot_hx2_steam_nom = pyo.Param(initialize=thermo_ref["M_dot_hx2_steam_nom"])
    model.M_dot_hx1_salt_nom  = pyo.Param(initialize=thermo_ref["M_dot_hx1_salt_nom"])
    model.M_dot_hx2_water_nom = pyo.Param(initialize=thermo_ref["M_dot_hx2_water_nom"])
    
    
    # hx conversion factors
    model.K_hx1               = pyo.Param(initialize=thermo_ref["K_hx1"])
    model.K_hx2               = pyo.Param(initialize=thermo_ref["K_hx2"])
    
    
    # mass to power conversions
    model.K_power             = pyo.Param(initialize=thermo_ref["K_power"])
    model.K_loss              = pyo.Param(initialize=thermo_ref["K_loss"])
    
    
    # steam extraction limit
    model.F_ext               = pyo.Param(initialize=thermo_ref["F_ext"])
    
    
    # installed capacity calculataions
    W_dot_inst                = oversize_factor * thermo_ref["W_dot_ref"]
    
    
    # min & max power cycle bounds
    model.W_dot_inst          = pyo.Param(initialize=W_dot_inst)
    model.W_dot_max           = pyo.Param(initialize=W_dot_inst)
    model.W_dot_min           = pyo.Param(initialize=0.25*W_dot_inst)
    model.W_dot_ramp_max      = pyo.Param(initialize=0.1*W_dot_inst)
    
    
    # steam mass flow bounds
    model.M_dot_hpt_max       = pyo.Param(initialize=(W_dot_inst/thermo_ref["W_dot_ref"])*thermo_ref["M_dot_hx1_steam_nom"])
    model.M_dot_hpt_min       = pyo.Param(initialize=(0.25*W_dot_inst/thermo_ref["W_dot_ref"])*thermo_ref["M_dot_hx1_steam_nom"])
    
    
    # other parameters
    model.Cp_salt             = pyo.Param(initialize=thermo_ref["Cp_salt"])
    model.Delta_T_hx1_salt    = pyo.Param(initialize=thermo_ref["Delta_T_hx1_salt"])
    model.Delta_H_hx2_water   = pyo.Param(initialize=thermo_ref["Delta_H_hx2_water"])

    
    # salt flow from reactor
    model.M_dot_salt          = pyo.Param(model.T, initialize={t:thermo_ref["M_dot_hx1_salt_nom"] for t in model.T})
    
      
    
  
    
def add_process_params(
       model,
       process_scale=None):
       

       process_scale = process_scale or {}

       def scale(val, key):
           return val * process_scale.get(key, 1.0)
       

       # seawater composition (units: kg/m^3)
       model.Conc_sw = pyo.Param(model.ions, initialize={'Li': 0.00018, 'Na': 10.8, 'Cl': 19.3})
       
       
       # base electricity requirements (units: kWh-e/m^3)
       elec_base = {
           'RO':  4.5,
           'MED': 1.0,
           'ED':  3.5,
           'CRY': 3.5}
       
       
       # base heat requirements (units:kWht-h/m^3)
       heat_base = {
           'RO':  0.0,
           'MED': 25.0,
           'ED':  0.0,
           'CRY': 90.0}
       
       
       # capex base cost for each process
       capex_base = {
           'RO':  500.0,
           'MED': 900.0,
           'ED':  200.0,
           'CRY': 950.0}
       

       # electricity requirements, sweepable
       model.Elec_required = pyo.Param(model.processes, initialize={p:scale(elec_base[p], f"{p}_elec") for p in model.processes})
       
       
       # heat requirements, sweepable
       model.Q_required = pyo.Param(model.processes, initialize={p:scale(heat_base[p], f"{p}_heat") for p in model.processes})
       
       
       # capex requirements, sweeapable
       model.K_process = pyo.Param(model.processes, initialize={p:scale(capex_base[p], f"{p}_capex") for p in model.processes})

       
        
   
 
# cost and price parameters

def add_economic_params(model, price_scale=None, cost_scale=None):
    

    # default scaling - no change if scale not specified
    price_scale = price_scale or {}
    cost_scale  = cost_scale or {}

    def scale(val, key, scale_dict):
        return val * scale_dict.get(key, 1.0)


    # base prices
    price_base    = {
        "water":        1.0,      # units: $/m^3
        "lithium":      70.0      # units: $/kg
    }


    # base costs
    cost_base     = {
        "ramp_power":   0.044,    # units: $/kWe 
        "tes_hot":      29.8,     # units: $/kWh-th
        "tes_cold":     20.0      # units: $/kWh-th
    }


    # commodity prices, scaled independently
    model.Price_water     = pyo.Param(initialize=scale(price_base["water"], "water", price_scale))
    model.Price_li        = pyo.Param(initialize=scale(price_base["lithium"], "lithium", price_scale))
    model.Price_salt      = pyo.Param(initialize=0.0)


    # power ramping cost scales with installed capacity 
    model.Cost_ramp_power = pyo.Param(initialize=scale(cost_base["ramp_power"], "ramp_power", cost_scale)*model.W_dot_inst.value/model.M_dot_hx1_steam_nom.value)


    # tes costs, converted from $/kWh to $/kg
    model.Cost_tes_hot    = pyo.Param(initialize=scale(cost_base["tes_hot"], "tes_hot", cost_scale)/3600*model.Cp_salt.value*model.Delta_T_hx1_salt.value)    
    model.Cost_tes_cold   = pyo.Param(initialize=scale(cost_base["tes_cold"], "tes_cold", cost_scale)/3600*model.Delta_H_hx2_water.value)


    # financial assumptions
    model.discount_rate   = pyo.Param(initialize=0.07)
    model.inflation_rate  = pyo.Param(initialize=0.025)
    model.lifetime        = pyo.Param(initialize=30)
    
    
    # capital recovery factor calculation
    def crf_init(model):
        f_prime = (1 + model.discount_rate)/(1 + model.inflation_rate) - 1
        return f_prime * (1 + f_prime) ** model.lifetime / ((1 + f_prime) ** model.lifetime - 1)
    
    model.capital_recovery_factor = pyo.Param(initialize=crf_init)


def add_zld_params(model, yield_scale=1.0):
    
    
    # time parameters
    model.Lifetime           = pyo.Param(initialize=30.0)             # units: yrs,          amortization time 
    model.Convert_time       = pyo.Param(initialize=3600.0)           # units: s/h,          conversion between seconds and hours
    model.Time_day           = pyo.Param(initialize=24)               # units: hr/day        number of hours in day


    # power curve parameters
    model.Slope_power_curve  = pyo.Param(initialize=1.075)            # units: --,           slope of power curve   
    model.Interc_power_curve = pyo.Param(initialize=0.0748)           # units: --,           intercept of power curve
    model.F_lpt_post         = pyo.Param(initialize=0.186)            # units: --,           fraction of power produced past extraction point in power cycle

    
    # recovery ratios & zld parameters
    model.Recovery_ro        = pyo.Param(initialize=0.50)             # units: --,           recovery ratio for ro
    model.Recovery_sw_to_med = pyo.Param(initialize=0.77)             # units: --,           recovery ratio for med fed from seawater
    model.Recovery_ro_to_med = pyo.Param(initialize=0.53)             # units: --,           recovery ratio for med fed from ro brine
    model.Ratio_na_to_li     = pyo.Param(initialize=5.0)              # units: --,           ratio needed for precipitation of lithium
    
    
    # big m parameter
    model.V_dot_max          = pyo.Param(initialize=40000.0)
    

    # ed yield, sweepable
    model.yield_ed_li        = pyo.Param(initialize=yield_scale*1.0)  # units: --,            yield for lithium from ed
    
    
    # little m parameter
    model.V_dot_min          = pyo.Param(initialize=1e-3)

    # calculate m for each ion
    def m_ion_init(model, i):
        return pyo.value(model.V_dot_max) * 20.0 * pyo.value(model.Conc_sw[i])
    
    model.M_ion              = pyo.Param(model.ions, initialize=m_ion_init)


    
    
