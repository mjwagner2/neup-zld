# import necessary packages
import pyomo.environ as pyo



"""TEST CASES


Base case: SW->MED. Note: extremely good (but not implausible) heat diversion from power conversion cycle is driving this being best
                    Note2: not fully MED, it maxes out at a cycle extraction tradeoff point, which is interesting and good

All perturbations return to nominal before making new change

Perturbation 1: Drop RO energy use to 3.0 - RO enters the system, MED is still present

Perturbation 2: 10x MED electricity use - RO only at maximum

Perturbation 3: water is worthless - no active links

Perturbation 4: lithium price x100. SW->MED->ED. Behavior is sensible

Perturbation 5: 10x MED electricity usage AND lithium price x100. SW->RO->ED

Perturbation 6: 100x MED electricity usage AND lithium price x100. SW->RO->ED. Now MED is not worth it

Perturbation 7: Salt is now worth $100/kg. Processed food becomes far too expensive. SW->RO->MED->CRY. No point in ED as Li is too cheap and it doesnt concentrate. But use CRY

Perturbation 8: Salt is worth $100/kg, Li price x100. Seawater mining takes off. RDO-3 team wins Nobel Prize for Engineering. SW->RO->MED->ED->CRY is built. $724Bn in revenue.

Perturbation 9: 10x MED electricity use, RO up to 4.6 electricity. No active links. This is important cos RO is v.marginal at baseline

"""

def main(Li_price=70,ED_elec_req=3.5,RO_elec_req=4.5,MED_elec_req=1.0):

    # build model
    
    model = pyo.ConcreteModel()
    


    # SETS 
    
    # set of processes available for use
    model.processes = pyo.Set(initialize=['RO', 'MED', 'ED', 'CRY'])
    
    
    # set of all possible links between processes
    model.links     = pyo.Set(dimen=2, initialize=[
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
    
    
    # set of all ions considered
    model.ions      = pyo.Set(initialize=['Li', 'Na', 'Cl'])
    
    
    # set of all power cycle configurations
    model.cycles    = pyo.Set(initialize=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'])
    
    
    ## PARAMETERS
    
    
    # recovery ratios []
    model.Recovery_ro         = pyo.Param(initialize=0.50)   
    model.Recovery_sw_to_med  = pyo.Param(initialize=0.77)   
    model.Recovery_ro_to_med  = pyo.Param(initialize=0.53)   
    model.Ratio_ed_na_li      = pyo.Param(initialize=5.0)    
    model.yield_ed_li         = pyo.Param(initialize=1.0)                 

    
    # commodity pricing [$/m3], [$/kWh], [$/kg], [$/kg]
    model.Price_water         = pyo.Param(initialize=1.0)    
    model.Price_elec          = pyo.Param(initialize=0.1)    
    model.Price_li            = pyo.Param(initialize=Li_price) 
    model.Price_salt          = pyo.Param(initialize=0.0)    
    
    
    # power cycle [kg/s], [kJ/kg], [kW/kg/s], [kW]
    model.M_dot_max           = pyo.Param(initialize=67.07)  
    model.H_extract           = pyo.Param(model.cycles, initialize={'C1': 2.82e3, 'C2': 2.79e3, 'C3': 2.49e3, 'C4': 2.38e3, 'C5': 2.32e3, 'C6': 2.21e3, 'C7': 2.20e3})  
    model.power_slope         = pyo.Param(model.cycles, initialize= {'C1': -873, 'C2': -829, 'C3': -540, 'C4': -427, 'C5': -335, 'C6': -229, 'C7': -211})
    model.power_intercept     = pyo.Param(initialize=52920)
    
    
    # big-M parameters
    model.V_dot_max           = pyo.Param(initialize=40000)
    model.V_dot_min           = pyo.Param(initialize=1E-04)
    model.Q_dot_max           = pyo.Param(initialize=2E05)
    
    
    # capex costs [$/kw], [$/m3/d]
    model.K_cycle             = pyo.Param(initialize=6000)   
    model.K_process           = pyo.Param(model.processes, initialize={'RO': 500.0,'MED': 900.0,'ED': 200.0, 'CRY': 950.0})

    
    # time conversions [s/h], [h/yr], [h/d], [yr]
    model.T_hour_to_sec       = pyo.Param(initialize=3600)   
    model.T_year_to_hour      = pyo.Param(initialize=8760)
    model.T_day_to_hour       = pyo.Param(initialize=24)
    model.T_amoritization     = pyo.Param(initialize=30)
    

    # seawater concentration [kg/m3]
    model.Conc_sw             = pyo.Param(model.ions, initialize={'Li': 0.00018, 'Na': 10.8, 'Cl': 19.3})                                                               # units: kg/m3     mass concentration of ions in seawater


    # big-M for concentration [kg/m3]
    def m_ion_init(model, i):
        return pyo.value(model.V_dot_max) * 5.0 * pyo.value(model.Conc_sw[i])
    model.M_ion = pyo.Param(model.ions, initialize=m_ion_init)
    
        
    # energy consumption [$/kWhe], [$/kWhth]
    model.Elec_required       = pyo.Param(model.processes, initialize={'RO': RO_elec_req, 'MED': MED_elec_req, 'ED': ED_elec_req,'CRY': 3.5})                                                 
    model.Q_required          = pyo.Param(model.processes, initialize={'RO': 0.0,  'MED': 25.0 ,'ED': 0.0, 'CRY': 90})                                                 


    
    # VARIABLES
    
    # binary variables
    model.y_link_active       = pyo.Var(model.links, within=pyo.Binary)                              
    model.y_process_active    = pyo.Var(model.processes, within=pyo.Binary)                          
    model.y_cycle_active      = pyo.Var(model.cycles, within=pyo.Binary)                             
    model.y_q_source          = pyo.Var(model.cycles, model.processes, within=pyo.Binary)            
    
    
    # volumetric flow [m3/h]
    model.v_dot_in            = pyo.Var(model.processes, within=pyo.NonNegativeReals)                
    model.v_dot_conc          = pyo.Var(model.processes, within=pyo.NonNegativeReals)                
    model.v_dot_dil           = pyo.Var(model.processes, within=pyo.NonNegativeReals)                
    model.v_dot_sw_feed       = pyo.Var(model.processes, within=pyo.NonNegativeReals)               
    model.v_dot_link          = pyo.Var(model.links, within=pyo.NonNegativeReals)                    
    
    
    # power cycle mass flows [kg/s]
    model.m_dot_cycle         = pyo.Var(model.cycles, bounds=(0, model.M_dot_max))                    
    model.m_dot_extract       = pyo.Var(model.cycles, bounds=(0, 0.8*model.M_dot_max))               
    
    
    # electricity [kWh-e]
    model.elec_used           = pyo.Var(model.processes, within=pyo.NonNegativeReals)                
    model.elec_generated      = pyo.Var(model.cycles, within=pyo.NonNegativeReals)                   
    model.elec_sold           = pyo.Var(model.cycles, within=pyo.NonNegativeReals)                     
    model.elec_allocated      = pyo.Var(model.cycles, model.processes, within=pyo.NonNegativeReals)  
    
    
    # heat [kWh-th]
    model.q_generated         = pyo.Var(model.cycles, within=pyo.NonNegativeReals)                   
    model.q_used              = pyo.Var(model.processes, within=pyo.NonNegativeReals)                
    model.q_allocated         = pyo.Var(model.cycles, model.processes, within=pyo.NonNegativeReals)  
    
    
    # ion flows [kg/h]
    model.f_ion_in            = pyo.Var(model.processes, model.ions, within=pyo.NonNegativeReals)    
    model.f_ion_conc          = pyo.Var(model.processes, model.ions, within=pyo.NonNegativeReals)
    model.f_ion_dil           = pyo.Var(model.processes, model.ions, within=pyo.NonNegativeReals)
    model.f_ion_link          = pyo.Var(model.links, model.ions, within=pyo.NonNegativeReals)
    model.f_li_recovered      = pyo.Var(within=pyo.NonNegativeReals)
    
    
    # capacity variables [m3/d], [kWe]
    model.v_dot_cap           = pyo.Var(model.processes, within=pyo.NonNegativeReals)
    model.w_dot_cap           = pyo.Var(model.cycles, within=pyo.NonNegativeReals)


    
    """BL: I believe this constraints are good but are messing up the solver. Instead have fixed this to 15 which 
    I believe roughly lines up with the exit salinity of MED (or for that matter ED). I believe this is OK
    because RO->CRY will never make sense, and all we are doing is slightly penalizing that
    I cannot resolve the numbers from the paper cited with that for MED and have instead used a rule of thumb from online
    that crystallizer needs about 4x as much heat and 3x as much power. This needs revisiting
  """



    def link_used_if_active(m, p, q):
        return m.v_dot_link[(p,q)] >= m.V_dot_min * m.y_link_active[(p,q)]
    model.link_used_if_active = pyo.Constraint(model.links, rule=link_used_if_active)


  
    def ion_link_off(m, p, q, i):
        if p == 'SW':
            return pyo.Constraint.Skip
        return m.f_ion_link[(p,q), i] <= m.M_ion[i] * m.y_link_active[(p,q)]
    model.ion_link_off = pyo.Constraint(model.links, model.ions, rule=ion_link_off)


  
    def linear_power_generation_rule(m,c):
        return m.elec_generated[c] == (m.power_slope[c] * m.m_dot_extract[c]  + m.power_intercept) * model.y_cycle_active[c]
    model.linear_power_generation = pyo.Constraint(model.cycles, rule=linear_power_generation_rule)
    
  
    
    def ion_conservation(m, p, i):
        return m.f_ion_in[p, i] == m.f_ion_conc[p, i] + m.f_ion_dil[p, i]
    model.ion_conservation = pyo.Constraint(model.processes, model.ions, rule=ion_conservation)
    
    
    
    def ion_mass_in(m, p, i):
        return m.f_ion_in[p, i] == sum(m.f_ion_link[(q,p), i] for (q,r) in m.links if r == p)
    model.ion_mass_in = pyo.Constraint(model.processes, model.ions, rule=ion_mass_in)



    def ed_precip_ratio(m):
        return m.f_ion_dil['ED', 'Na'] == m.Ratio_ed_na_li * m.f_ion_dil['ED', 'Li']
    model.ed_precip_ratio = pyo.Constraint(rule=ed_precip_ratio)
    
    
    
    def li_recovery(m):
        return m.f_li_recovered == m.yield_ed_li * m.f_ion_in['ED', 'Li']
    model.li_recovery = pyo.Constraint(rule=li_recovery)
    
    
    
    def li_mass_balance(m):
        return m.f_li_recovered == m.f_ion_dil['ED', 'Li']
    model.li_mass_balance = pyo.Constraint(rule=li_mass_balance)
    
    

    def ed_na_split(m):
        return (m.f_ion_conc['ED', 'Na'] * m.v_dot_dil['ED'] == m.f_ion_dil['ED', 'Na'] * m.v_dot_conc['ED'])
    model.ed_na_split = pyo.Constraint(rule=ed_na_split)
    
    
    
    def ed_cl_split(m):
        return (m.f_ion_conc['ED', 'Cl'] * m.v_dot_dil['ED'] == m.f_ion_dil['ED', 'Cl'] * m.v_dot_conc['ED'])
    model.ed_cl_split = pyo.Constraint(rule=ed_cl_split)



    def ion_link_gate(model, p,q,i):
        M = pyo.value(model.Conc_sw[i]) * pyo.value(model.V_dot_max)
        return model.f_ion_link[(p,q), i] <= M * model.y_link_active[(p,q)]
    model.ion_link_gate = pyo.Constraint(model.links, model.ions, rule=ion_link_gate)
    


    def ion_link_upper(model,p,q,i):
        if p == 'SW': 
            return pyo.Constraint.Skip
        return model.f_ion_link[(p,q),i] <= model.f_ion_conc[p,i]
    model.ion_link_upper  = pyo.Constraint(model.links, model.ions, rule=ion_link_upper)
    


    def ion_link_lower(model,p,q,i):
        if p == 'SW': 
            return pyo.Constraint.Skip
        return model.f_ion_link[(p,q),i] >= model.f_ion_conc[p,i] - model.M_ion[i] * (1 - model.y_link_active[(p,q)])
    model.ion_link_lower  = pyo.Constraint(model.links, model.ions, rule=ion_link_lower)
    


    def process_requires_incoming(model, p):
        if p == 'SW':
            return pyo.Constraint.Skip
        incoming = [u for (u,v) in model.links if v == p]
        if not incoming:
            return pyo.Constraint.Skip
        return model.y_process_active[p] <= sum(model.y_link_active[u,p] for u in incoming)
    model.process_requires_incoming = pyo.Constraint(model.processes, rule=process_requires_incoming)
    


    def sw_ion_link(m,p,i):
        if ('SW',p) not in m.links:
            return pyo.Constraint.Skip
        return m.f_ion_link[('SW',p),i] == m.v_dot_link[('SW',p)] * m.Conc_sw[i]
    model.sw_ion_link = pyo.Constraint(model.processes, model.ions, rule=sw_ion_link)
    
    
    
    def heat_generated_rule(m, c):
        return m.q_generated[c] == m.H_extract[c] * m.m_dot_extract[c]
    model.heat_generated = pyo.Constraint(model.cycles, rule=heat_generated_rule)
    
     

    def heat_allocated_capacity_rule(m, c):
        return sum(m.q_allocated[c, p] for p in m.processes) == m.q_generated[c]
    model.heat_allocated_capacity = pyo.Constraint(model.cycles, rule=heat_allocated_capacity_rule)
    
    

    def heat_used_balance_rule(m, p):
        return m.q_used[p] == sum(m.q_allocated[c, p] for c in m.cycles)
    model.heat_used_balance = pyo.Constraint(model.processes, rule=heat_used_balance_rule)            
    
    

    def heat_gating_rule(m, c, p):
        return m.q_allocated[c, p] <= m.Q_dot_max * m.y_q_source[c,p]
    model.heat_gating = pyo.Constraint(model.cycles, model.processes, rule=heat_gating_rule)
    

    
    def heat_used_by_process(m, p):
        return m.q_used[p] == m.Q_required[p] * m.v_dot_in[p]
    model.heat_used_by_process = pyo.Constraint(model.processes, rule=heat_used_by_process)
    

    
    def electricity_balance_rule(m, c):
        return m.elec_sold[c] == m.elec_generated[c] - sum(m.elec_allocated[c, p] for p in m.processes)
    model.electricity_balance = pyo.Constraint(model.cycles, rule=electricity_balance_rule)
    
    

    def electric_used_by_process(m, p):
        return m.elec_used[p] == m.Elec_required[p] * m.v_dot_in[p]
    model.electric_used_by_process = pyo.Constraint(model.processes, rule=electric_used_by_process)
    
    

    def electricity_used_balance_rule(m, p):
        return m.elec_used[p] == sum(m.elec_allocated[c, p] for c in m.cycles)
    model.electricity_used_balance = pyo.Constraint(model.processes, rule=electricity_used_balance_rule)

    
    
    def seawater_intake(m):
        return sum(m.v_dot_link['SW', p] for (upstream, p) in m.links if upstream == 'SW') <= m.V_dot_max
    model.seawater_intake = pyo.Constraint(rule=seawater_intake)
    
    

    def split_flow_rule(m, c):
        return m.m_dot_cycle[c] + m.m_dot_extract[c] == m.M_dot_max * model.y_cycle_active[c]
    model.split_flow = pyo.Constraint(model.cycles, rule=split_flow_rule)
    
    

    def process_flow_balance(m, p):
        return m.v_dot_in[p] == m.v_dot_conc[p] + m.v_dot_dil[p]
    model.process_flow_balance = pyo.Constraint(model.processes, rule=process_flow_balance)
    
    

    def inlet_flow(m, p):
        inflow = sum(m.v_dot_link[(p2,p)] for (p2,q) in m.links if q==p)
        return m.v_dot_in[p] == inflow
    model.inlet_flow = pyo.Constraint(model.processes, rule=inlet_flow)
    
    
    
    def link_gate_all(m, p, q):
        return m.v_dot_link[(p,q)] <= m.V_dot_max * m.y_link_active[(p,q)]
    model.link_gate_all = pyo.Constraint(model.links, rule=link_gate_all)

    

    def process_used_if_active(model,p):
        return model.v_dot_in[p] >= model.V_dot_min * model.y_process_active[p]
    model.process_used_if_active = pyo.Constraint(model.processes, rule=process_used_if_active)



    def inlet_flow_max(model, p):
        return model.v_dot_in[p] <= model.V_dot_max * model.y_process_active[p]
    model.inlet_flow_max = pyo.Constraint(model.processes, rule=inlet_flow_max)
     


    def single_sw_feed(model):
        return sum(model.y_link_active['SW', p] for p in model.processes if ('SW', p) in model.links) <= 1
    model.single_sw_feed = pyo.Constraint(rule=single_sw_feed)
    


    def single_upstream(model, q):
        return sum(model.y_link_active[(p2,q)] for (p2,q2) in model.links if q2==q) <= 1
    model.single_upstream = pyo.Constraint(model.processes, rule=single_upstream)
    


    def single_downstream(model, p):
        terms = [model.y_link_active[(p,q)] for (p2,q) in model.links if p2 == p]
        if not terms:
            return pyo.Constraint.Skip
        return sum(terms) <= 1
    model.single_downstream = pyo.Constraint(model.processes, rule=single_downstream)
    


    def upstream_activation(model, q, p):
        if q == 'SW':
            return pyo.Constraint.Skip
        return model.y_link_active[(q,p)] <= model.y_process_active[q]
    model.upstream_activation = pyo.Constraint(model.links, rule=upstream_activation)
  
    

    def downstream_activation(model, q, p):
        return model.y_link_active[q,p] <= model.y_process_active[p]
    model.downstream_activation = pyo.Constraint(model.links, rule=downstream_activation)
    


    def upper_link_flow_balance(m, p, q):
        if p == 'SW':
            return pyo.Constraint.Skip
        return m.v_dot_link[p,q] <= m.v_dot_conc[p]
    model.upper_link_flow_balance = pyo.Constraint(model.links, rule=upper_link_flow_balance)
    
    

    def lower_link_flow_balance(model,p,q):
        if p == 'SW':
            return pyo.Constraint.Skip
        return model.v_dot_link[(p,q)] >= model.v_dot_conc[p] - model.V_dot_max*(1 - model.y_link_active[(p,q)])
    model.lower_link_flow_balance = pyo.Constraint(model.links, rule=lower_link_flow_balance)
    
    
    
    def recovery_ro(m):
        return m.v_dot_dil['RO'] == m.Recovery_ro * m.v_dot_in['RO']
    model.recovery_ro = pyo.Constraint(rule=recovery_ro)
    
    

    def recovery_med(m):
        return m.v_dot_dil['MED'] == (
            m.v_dot_in['MED'] * m.Recovery_sw_to_med * (1 - m.y_link_active['RO','MED']) + 
            m.v_dot_in['MED'] * m.Recovery_ro_to_med * m.y_link_active['RO','MED'])
    model.recovery_med = pyo.Constraint(rule=recovery_med)
    
    

    def diluted_ion(m, p, i):
        if p == 'ED':
            return pyo.Constraint.Skip
        return m.f_ion_dil[p, i] == 0.0
    model.diluted_ion = pyo.Constraint(model.processes, model.ions, rule=diluted_ion)

    
        
    def cry_no_concentrate(m):
        return m.v_dot_conc['CRY'] == 0.0
    model.cry_no_concentrate = pyo.Constraint(rule=cry_no_concentrate)
    
    
    
    def process_capacity_limit(m, p):
        return m.v_dot_in[p] * m.T_day_to_hour <= m.v_dot_cap[p]
    model.process_capacity_limit = pyo.Constraint(model.processes, rule=process_capacity_limit)
    
    
       
    def process_capacity_activation(m, p):
        return m.v_dot_cap[p] <= m.V_dot_max * m.T_day_to_hour * m.y_process_active[p]
    model.process_capacity_activation = pyo.Constraint(model.processes, rule=process_capacity_activation)



    def cycle_capacity_limit(m, c):
        return m.elec_generated[c] <= m.w_dot_cap[c]
    model.cycle_capacity_limit = pyo.Constraint(model.cycles, rule=cycle_capacity_limit)




    # OBJECTIVE FUNCTION
    
    profit = model.T_year_to_hour * (
        model.Price_li * model.f_li_recovered + 
        model.Price_water * (model.v_dot_dil['RO'] + model.v_dot_dil['MED'] + model.v_dot_dil['CRY']) +
        model.Price_elec * sum(model.elec_sold[c] for c in model.cycles) +
        model.Price_salt * sum(model.f_ion_in['CRY', i] for i in model.ions))

    cost = (
        sum(model.K_cycle * model.w_dot_cap[c] for c in model.cycles) / model.T_amoritization
        + sum(model.K_process[p] * model.v_dot_cap[p] for p in model.processes) / model.T_amoritization)


    model.obj = pyo.Objective(expr=(profit-cost), sense=pyo.maximize)
    
    
    solver = pyo.SolverFactory("gurobi")
    solver.options['NonConvex'] = 2
    results = solver.solve(model, tee=False)
    
    
    def safe_val(v, fmt="{:.2f}"):
        """Return variable value or 'not solved' if uninitialized."""
        if v.value is None:
            return "not solved"
        try:
            return fmt.format(pyo.value(v))
        except:
            return v.value
    
    def nval(x):
        try:
            v = pyo.value(x)
            return None if v is None else float(v)
        except Exception:
            return None
    
    def fmt_flow(v):
        return "not solved" if v is None else f"{v:.2f} m³/h"
    
    def fmt_mass(v):
        return "not solved" if v is None else f"{v:.2f} kg/h"
    
    def fmt_conc(v):
        if v is None:
            return "not solved"
        mgL = v * 1000.0
        return f"{v:.6f} kg/m³ ({mgL:.1f} mg/L)"
    
    def fmt_resid(v):
        if v is None:
            return "n/a"
        tag = "" if abs(v) <= 1e-6 else "  <-- check"
        return f"{v:+.6e} kg/h{tag}"
    
    if __name__ == "__main__":
    
        print(f" li_yield = {safe_val(model.yield_ed_li)}")
        
        
        print("\n================== POWER CYCLE RESULTS ==================")
        for c in model.cycles:
            print(f"\nCycle {c}:")
            print(f"  ṁ_cycle_unit        = {safe_val(model.m_dot_cycle[c])} kg/s")
            print(f"  ṁ_extract_unit      = {safe_val(model.m_dot_extract[c])} kg/s")
            print(f"  Ẇ_generated_unit    = {safe_val(model.elec_generated[c])} kWe")
            print(f"  Ẇ_sold_unit         = {safe_val(model.elec_sold[c])} kWe")
            print(f"  Q_generated         = {safe_val(model.q_generated[c])} kWth")
        
        
        print("\n================== PROCESS RESULTS ==================")
        for p in model.processes:
            print(f"\nProcess {p}:")
            print(f"  Active?             = {safe_val(model.y_process_active[p], '{:.0f}')}")
            print(f"  v̇_in                = {safe_val(model.v_dot_in[p])} m³/h")
            print(f"  v̇_conc              = {safe_val(model.v_dot_conc[p])} m³/h")
            print(f"  v̇_dil               = {safe_val(model.v_dot_dil[p])} m³/h")
            print(f"  Electricity used    = {safe_val(model.elec_used[p])} kWe")
            print(f"  Heat used           = {safe_val(model.q_used[p])} kW-th")
        
            for i in model.ions:
                print(f"    Ion {i}:")
                print(f"      f_in            = {safe_val(model.f_ion_in[p, i])} kg/h")
                print(f"      f_conc          = {fmt_conc(nval(model.f_ion_conc[p, i]))}")
                print(f"      f_dil           = {fmt_conc(nval(model.f_ion_dil[p, i]))}")
        
        
        print("\n================== ACTIVE LINKS ==================")
        active = False
        for (q, p) in model.links:
            y = nval(model.y_link_active[q, p])
            v = nval(model.v_dot_link[q, p])
            if y is not None and (y > 0.5 or (v is not None and v > 1e-3)):
                print(f"  {q} → {p}:  y_active={int(round(y))}, v̇ = {v:.2f} m³/h")
                active = True
        if not active:
            print("  (No active links)")
        
        
        print("\n================== OBJECTIVE VALUE ==================")
        try:
            print(f"  Profit = {pyo.value(model.obj):,.2f} $")
        except:
            print("  Profit = not solved")
    
    return(float(safe_val(model.f_li_recovered)),float(safe_val(model.v_dot_in['MED'])),float(safe_val(model.v_dot_in['RO'])))

if __name__ == "__main__":
    main()


