import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from utils import load_timeseries, get_cap_cost, get_nodal_inputs, get_connection_info, annualization_rate

# both dry season 5-day model and annual model would use this nodes results function
def node_results_retrieval(args, m, i, T, nodal_load_input, config):
    # prepared info inputs
    if T == 216:
        dome_load_hourly_kw, solar_po_hourly, rain_rate_daily_mm_m2 = load_timeseries(args, mod_level="cap")
    elif T == 2160:
        dome_load_hourly_kw, solar_po_hourly, rain_rate_daily_mm_m2 = load_timeseries(args, mod_level="ope")
    if config == 1:
        dome_load = dome_load_hourly_kw / 100 * nodal_load_input["domestic_load_kwh_day"][i]
    elif config == 2:
        dome_load = dome_load_hourly_kw / 100 * np.sum(nodal_load_input["domestic_load_kwh_day"])

    node_df = pd.DataFrame()
    node_df['node_id'] = [i]
    node_df['solar_cap_kw'] = [m.getVarByName('solar_cap').X]
    node_df['diesel_cap_kw'] = [m.getVarByName('diesel_cap').X]
    node_df['batt_la_energy_cap_kwh'] = [m.getVarByName('batt_la_energy_cap').X]
    node_df['batt_la_power_cap_kw']   = [m.getVarByName('batt_la_power_cap').X]
    node_df['batt_li_energy_cap_kwh'] = [m.getVarByName('batt_li_energy_cap').X]
    node_df['batt_li_power_cap_kw']   = [m.getVarByName('batt_li_power_cap').X]

    node_ts_ar = np.zeros((T,11))
    for j in range(T):
        node_ts_ar[j,0] = m.getVarByName('solar_util[{}]'.format(j)).X
        node_ts_ar[j,1] = m.getVarByName('diesel_gen[{}]'.format(j)).X
        node_ts_ar[j,2] = m.getVarByName('batt_la_level[{}]'.format(j)).X
        node_ts_ar[j,3] = m.getVarByName('batt_la_charge[{}]'.format(j)).X
        node_ts_ar[j,4] = m.getVarByName('batt_la_discharge[{}]'.format(j)).X
        node_ts_ar[j,5] = m.getVarByName('batt_li_level[{}]'.format(j)).X
        node_ts_ar[j,6] = m.getVarByName('batt_li_charge[{}]'.format(j)).X
        node_ts_ar[j,7] = m.getVarByName('batt_li_discharge[{}]'.format(j)).X
        node_ts_ar[j,8] = m.getVarByName('irrigation_load[{}]'.format(j)).X
        node_ts_ar[j,9] = m.getVarByName('commercial_load[{}]'.format(j)).X
    node_ts_ar[:,10] = dome_load[0:T]

    return node_df, node_ts_ar


def system_ts_sum(ts_results):
    system_ts = np.sum(ts_results, axis=2)
    ts_col_names = ['solar_util_kw', 'diesel_util_kw', 'batt_la_level_kwh', 'batt_la_charge_kw', 'batt_la_discharge_kw',
                    'batt_li_level_kwh', 'batt_li_charge_kw', 'batt_li_discharge_kw',
                    'irrigation_load_kw', 'commercial_load_kw', 'domestic_load_kw', ]
    system_ts_df = pd.DataFrame(system_ts, columns=ts_col_names)
    return system_ts_df

def get_irrigation_ts(args, m, day_start, day_end):
    dome_load_hourly_kw, solar_po_hourly, rain_rate_daily_mm_m2 = load_timeseries(args, mod_level='ope')
    # get the daily irrigation time series
    daily_ts_ar = np.zeros(((day_end-day_start+1),4))
    daily_ts_ar[:,0] = rain_rate_daily_mm_m2[day_start:(day_end+1)]
    for d in range(day_end-day_start+1):
        daily_ts_ar[d,1] = m.getVarByName('ground_water_level_mm[{}]'.format(d)).X
        daily_ts_ar[d,2] = m.getVarByName('ground_water_charge_mm[{}]'.format(d)).X
        daily_ts_ar[d,3] = m.getVarByName('ground_water_discharge_mm[{}]'.format(d)).X
    irrigation_daily_ts_results = pd.DataFrame(daily_ts_ar, columns=['rain_rate_mm', 'ground_water_level_mm',
                                                                     'ground_water_charge_mm', 'ground_water_discharge_mm'])
    return irrigation_daily_ts_results

### --- this function would only be used by annul model --- ###
def process_results(args, nodes_results, system_ts_results, nodes_capacity_results, config):

    # Retrieve necessary model parameters
    T = args.num_hour_ope
    #num_nodes, irrigation_area_m2 = get_nodes_area(args, sce_sf_area_m2)
    dome_load_hourly_kw, solar_pot_hourly, rain_rate_daily_mm_m2 = load_timeseries(args, mod_level="ope")
    lv_connect_len, mv_connect_len, tx_num, total_tx_cost = tx_results(args, config)

    nodal_load_input = get_nodal_inputs(args)
    if config == 1:
        num_nodes = len(nodal_load_input)
    elif config == 2 or config == 3:
        num_nodes = 1
    irrigation_area_m2 = np.sum(nodal_load_input["irrigation_area_ha"]) * 1e4
    dome_load = dome_load_hourly_kw / 100 * np.sum(nodal_load_input["domestic_load_kwh_day"])

    # Calculate demand, generation, solar uncurtailed/actual CF
    avg_total_demand     = np.mean(system_ts_results.domestic_load_kw) + \
                           np.mean(system_ts_results.irrigation_load_kw) + \
                           np.mean(system_ts_results.commercial_load_kw)
    peak_total_demand    = np.max(system_ts_results.domestic_load_kw + system_ts_results.irrigation_load_kw +
                                  system_ts_results.commercial_load_kw)
    avg_solar_gen        = np.mean(system_ts_results.solar_util_kw)
    avg_diesel_gen       = np.mean(system_ts_results.diesel_util_kw)
    avg_total_gen        = avg_solar_gen + avg_diesel_gen
    solar_uncurtailed_cf = np.mean(solar_pot_hourly)
    solar_actual_cf      = avg_solar_gen / np.sum(nodes_results.solar_cap_kw)

    # total capital cost and operation cost
    solar_cap_cost, solar_single_cap_cost, battery_la_cap_cost_kwh, battery_li_cap_cost_kwh, \
    battery_inverter_cap_cost_kw, diesel_cap_cost_kw = get_cap_cost(args, args.num_year_ope)
    solar_unit_price_interpld = interp1d(args.solar_pw_cap_kw, solar_cap_cost)
    solar_cost_node = np.zeros(num_nodes)
    for i in range(num_nodes):
        solar_cost_node[i] = solar_unit_price_interpld(nodes_results.solar_cap_kw[i])
    total_solar_cost = np.sum(solar_cost_node)
    # total_solar_cost = np.sum(nodes_results.solar_cap_kw) * solar_single_cap_cost
    total_diesel_cost  = np.sum(nodes_results.diesel_cap_kw) * diesel_cap_cost_kw
    total_battery_la_cost = np.sum(nodes_results.batt_la_energy_cap_kwh) * battery_la_cap_cost_kwh + \
                            np.sum(nodes_results.batt_la_power_cap_kw) * battery_inverter_cap_cost_kw
    total_battery_li_cost = np.sum(nodes_results.batt_li_energy_cap_kwh) * battery_li_cap_cost_kwh + \
                            np.sum(nodes_results.batt_li_power_cap_kw) * battery_inverter_cap_cost_kw
    total_diesel_fuel_cost = avg_diesel_gen * T * args.diesel_cost_liter * args.liter_per_kwh / args.diesel_eff

    total_gen_cost = total_solar_cost + total_battery_la_cost + total_battery_li_cost + \
                     total_diesel_cost + total_diesel_fuel_cost
    total_elec_cost = total_gen_cost + total_tx_cost

    # Create arrays to store energy output & costs
    data_for_export = pd.DataFrame()

    ## Populate data_for_export
    data_for_export['config'] = [config]
    data_for_export['nodes'] = [num_nodes]
    data_for_export['total_irrigation_area_ha'] = [np.sum(irrigation_area_m2)/1e4]

    data_for_export['solar_cap_kw'] = [np.sum(nodes_results.solar_cap_kw)]
    data_for_export['diesel_cap_kw'] = [np.sum(nodes_results.diesel_cap_kw)]
    data_for_export['diesel_cap_kw_in_ds_model'] = [np.sum(nodes_capacity_results.diesel_cap_kw)]
    data_for_export['battery_la_energy_cap_kwh'] = [np.sum(nodes_results.batt_la_energy_cap_kwh)]
    data_for_export['battery_la_power_cap_kw']   = [np.sum(nodes_results.batt_la_power_cap_kw)]
    data_for_export['battery_li_energy_cap_kwh'] = [np.sum(nodes_results.batt_li_energy_cap_kwh)]
    data_for_export['battery_li_power_cap_kw']   = [np.sum(nodes_results.batt_li_power_cap_kw)]
    data_for_export['MV_connect_wire_m'] = [mv_connect_len]
    data_for_export['LV_connect_wire_m'] = [lv_connect_len]
    # data_for_export['LV_dist_wire_m']    = [lv_dist_len]
    data_for_export['transformer_numbers'] = [tx_num]

    data_for_export['peak_load_kw'] = [peak_total_demand]
    data_for_export['avg_load_kw']  = [avg_total_demand]
    data_for_export['avg_gen_kw']   = [avg_total_gen]
    data_for_export['avg_solar_gen_kw']   = [avg_solar_gen]
    data_for_export['avg_diesel_gen_kw']  = [avg_diesel_gen]
    data_for_export['solar_unc_cf'] = [solar_uncurtailed_cf]
    data_for_export['solar_act_cf'] = [solar_actual_cf]

    data_for_export['solar_cost'] = [total_solar_cost]
    data_for_export['diesel_cost'] = [(total_diesel_cost + total_diesel_fuel_cost)]
    data_for_export['diesel_cap_cost'] = [total_diesel_cost]
    data_for_export['diesel_fuel_cost'] = [total_diesel_fuel_cost]
    data_for_export['battery_la_cost'] = [total_battery_la_cost]
    data_for_export['battery_li_cost'] = [total_battery_li_cost]
    data_for_export['connection_cost'] = [total_tx_cost]
    data_for_export['generation_cost'] = [total_gen_cost]
    data_for_export['electricity_cost'] = [total_elec_cost]

    data_for_export['LCOE'] = [total_elec_cost / (T*avg_total_demand)]

    return data_for_export


def tx_results(args, config):
    # connection wire
    if config == 1:
        lv_connect_len, mv_connect_len, tx_num = 0, 0, 0
    elif config == 2:
        lv_connect_len, mv_connect_len, tx_num = get_connection_info(args)
    else:
        lv_connect_len, mv_connect_len, tx_num = 0, 0, 0

    tx_ann_rate = annualization_rate(args.i_rate, args.annualize_years_trans)
    total_tx_cost = args.num_year_ope * tx_ann_rate * ((lv_connect_len) * float(args.trans_lv_cost_kw_m) +
                                                   mv_connect_len * float(args.trans_mv_cost_kw_m) +
                                                   tx_num * float(args.transformer_cost))
    return lv_connect_len, mv_connect_len, tx_num, total_tx_cost


