from gurobipy import *
from utils import get_cap_cost, load_timeseries, get_nodal_inputs, get_fixed_load
from results_processing import node_results_retrieval, system_ts_sum, process_results
import numpy as np
import pandas as pd
import os

# for the fixed model, there is only
def create_fix_load_model(args, scenario_name, config, lan_tlnd_out, scenario_start_time):
    print("fixed load model building and solving")
    print("--------####################------------")
    # Load timeseries data
    T = args.num_hour_fixed_load
    trange = range(T)
    solar_region = lan_tlnd_out["DistName"].lower()
    dome_load_hourly_kw, solar_po_hourly, rain_rate_daily_mm_m2 = load_timeseries(args, solar_region, mod_level="fixed_load")
    nodal_fixed_load = get_fixed_load(args)
    # Extract nodal load inputs
    nodal_load_input = get_nodal_inputs(args, lan_tlnd_out)
    # currently, let fixed load model can only apply config 2.
    num_nodes = 1

    # Retrieve capital prices for solar, battery, and diesel generators
    solar_cap_cost, solar_single_cap_cost, battery_la_cap_cost_kwh, battery_li_cap_cost_kwh, \
    battery_inverter_cap_cost_kw, diesel_cap_cost_kw = get_cap_cost(args, args.num_year_cap)

    # initialize results table for storing the nodal generator capacities from capacity_model
    nodes_results = pd.DataFrame()
    # specify the time series outputs array
    ts_results = np.zeros((T,11,num_nodes))

    # set up the nodal loop. Each iteration will calculate the capacities of each node.
    for i in range(num_nodes):
        m = Model("fixed_load_model_" + str(i))
        print('fixed load model node ' + str(i) + ' building and solving')

        # Initialize capacity variables
        solar_cap = m.addVar(name='solar_cap')
        solar_binary = m.addVar(name='solar_cap_binary', vtype=GRB.BINARY)
        m.setPWLObj(solar_cap, args.solar_pw_cap_kw, solar_cap_cost)
        diesel_cap = m.addVar(obj=diesel_cap_cost_kw, name='diesel_cap')
        diesel_binary = m.addVar(name='diesel_cap_binary', vtype=GRB.BINARY)
        battery_la_cap_kwh = m.addVar(obj=battery_la_cap_cost_kwh, name = 'batt_la_energy_cap')
        battery_la_cap_kw  = m.addVar(obj=battery_inverter_cap_cost_kw, name = 'batt_la_power_cap')
        battery_li_cap_kwh = m.addVar(obj=battery_li_cap_cost_kwh, name = 'batt_li_energy_cap')
        battery_li_cap_kw  = m.addVar(obj=battery_inverter_cap_cost_kw, name = 'batt_li_power_cap')

        # constraints for tech availability
        if not args.solar_ava:
            m.addConstr(solar_cap == 0)
        else:
            m.addConstr(solar_cap - args.solar_min_cap * solar_binary >= 0)
            m.addConstr(solar_cap * (1 - solar_binary) == 0)

        if not args.battery_la_ava:
            m.addConstr(battery_la_cap_kwh == 0)
        if not args.battery_li_ava:
            m.addConstr(battery_li_cap_kwh == 0)

        if not args.diesel_ava:
            m.addConstr(diesel_cap == 0)
        else:
            m.addConstr(diesel_cap - args.diesel_min_cap * diesel_binary >= 0)
            m.addConstr(diesel_cap * (1-diesel_binary) == 0)
            if args.diesel_vali_cond:
                m.addConstr(diesel_binary == 1)

        # battery capacity constraints
        m.addConstr(battery_la_cap_kwh * (1-args.battery_la_min_soc) * float(args.battery_la_p2e_ratio_range[0]) <=
                    battery_la_cap_kw)
        m.addConstr(battery_la_cap_kwh * (1-args.battery_la_min_soc) * float(args.battery_la_p2e_ratio_range[1]) >=
                    battery_la_cap_kw)
        m.addConstr(battery_li_cap_kwh * (1-args.battery_li_min_soc) * float(args.battery_li_p2e_ratio_range[0]) <=
                    battery_li_cap_kw)
        m.addConstr(battery_li_cap_kwh * (1-args.battery_li_min_soc) * float(args.battery_li_p2e_ratio_range[1]) >=
                    battery_li_cap_kw)
        m.update()

        # Initialize time-series variables
        solar_util      = m.addVars(trange, name = 'solar_util')
        battery_la_charge    = m.addVars(trange, obj = args.nominal_charge_discharge_cost_kwh,name= 'batt_la_charge')
        battery_la_discharge = m.addVars(trange, obj = args.nominal_charge_discharge_cost_kwh,name= 'batt_la_discharge')
        battery_la_level     = m.addVars(trange, name='batt_la_level')
        battery_li_charge    = m.addVars(trange, obj = args.nominal_charge_discharge_cost_kwh,name= 'batt_li_charge')
        battery_li_discharge = m.addVars(trange, obj = args.nominal_charge_discharge_cost_kwh,name= 'batt_li_discharge')
        battery_li_level     = m.addVars(trange, name='batt_li_level')
        diesel_kwh_fuel_cost = args.diesel_cost_liter * args.liter_per_kwh / args.diesel_eff
        diesel_gen = m.addVars(trange, obj=diesel_kwh_fuel_cost, name="diesel_gen")

        # create irrigation and commercial loads
        irrigation_load = m.addVars(trange, obj=args.irrigation_nominal_cost, name='irrigation_load')
        com_load = m.addVars(trange, obj=args.com_nominal_cost, name='commercial_load')
        m.update()


        # Add time-series Constraints
        for j in trange:
            # solar and diesel generation constraint
            m.addConstr(diesel_gen[j] <= diesel_cap)
            m.addConstr(solar_util[j] <= solar_cap * round(solar_po_hourly[j], 4))

            # Energy Balance
            m.addConstr(solar_util[j] + diesel_gen[j] - battery_la_charge[j] + battery_la_discharge[j] - \
                        battery_li_charge[j] + battery_li_discharge[j] == nodal_fixed_load[j])

            # Battery operation constraints
            m.addConstr(args.battery_la_eff * battery_la_charge[j] - battery_la_cap_kw <= 0)
            m.addConstr(battery_la_discharge[j] / args.battery_la_eff - battery_la_cap_kw <= 0)
            m.addConstr(battery_la_level[j] - battery_la_cap_kwh <= 0)
            m.addConstr(battery_la_level[j] - battery_la_cap_kwh * args.battery_la_min_soc >=0)

            m.addConstr(args.battery_li_eff * battery_li_charge[j] - battery_li_cap_kw <= 0)
            m.addConstr(battery_li_discharge[j] / args.battery_li_eff - battery_li_cap_kw <= 0)
            m.addConstr(battery_li_level[j] - battery_li_cap_kwh <= 0)
            m.addConstr(battery_li_level[j] - battery_li_cap_kwh * args.battery_li_min_soc >=0)

            ## Battery control
            if j == 0:
                m.addConstr(
                    battery_la_discharge[j] / args.battery_la_eff - args.battery_la_eff * battery_la_charge[j] ==
                    battery_la_level[T - 1] - battery_la_level[j])
                m.addConstr(
                    battery_li_discharge[j] / args.battery_li_eff - args.battery_li_eff * battery_li_charge[j] ==
                    battery_li_level[T - 1] - battery_li_level[j])
            else:
                m.addConstr(
                    battery_la_discharge[j] / args.battery_la_eff - args.battery_la_eff * battery_la_charge[j] ==
                    battery_la_level[j - 1] - battery_la_level[j])
                m.addConstr(
                    battery_li_discharge[j] / args.battery_li_eff - args.battery_li_eff * battery_li_charge[j] ==
                    battery_li_level[j - 1] - battery_li_level[j])

            m.addConstr(irrigation_load[j] == 0)
            m.addConstr(com_load[j] == 0)

            m.update()

        # Set model solver parameters
        m.setParam("FeasibilityTol", args.feasibility_tol)
        m.setParam("OptimalityTol", args.optimality_tol)
        m.setParam("Method", args.solver_method)
        m.setParam("OutputFlag", 1)
        # Solve the model

        m.optimize()

        ### ------------------------- Results Output ------------------------- ###

        # Retrieve results and process the model solution for next step
        single_node_results, single_node_ts_results = node_results_retrieval(args, m, i, T, nodal_load_input, config,
                                                                             solar_region)
        nodes_results = pd.concat([nodes_results, single_node_results])
        nodes_results = nodes_results.reset_index(drop=True)
        ts_results[:, :, i] = single_node_ts_results

    # save results / get final processed results
    scenario_dir = scenario_name
    if not os.path.exists(os.path.join(args.results_dir, scenario_dir)):
        os.makedirs(os.path.join(args.results_dir, scenario_dir))

    system_ts_results = system_ts_sum(ts_results)
    system_ts_results['fixed_load_kw'] = nodal_fixed_load

    nodes_capacity_results = pd.DataFrame({'diesel_cap_kw': [0]})
    processed_results = process_results(args, nodes_results, system_ts_results, nodes_capacity_results,
                                        config, lan_tlnd_out, scenario_start_time)

    nodes_results.round(decimals=3).to_csv(os.path.join(args.results_dir, scenario_dir, 'raw_results.csv'))
    system_ts_results.round(decimals=3).to_csv(os.path.join(args.results_dir, scenario_name, 'ts_results.csv'))
    processed_results.round(decimals=3).to_csv(os.path.join(args.results_dir, scenario_name, 'processed_results.csv'))

    return None