from capacity_model import create_capacity_model
from operation_model import create_operation_model
# from fixed_load_model import create_fix_load_model
from utils import get_args
import datetime
import pandas as pd
import os

if __name__ == '__main__':

    running_start_time = datetime.datetime.now()

    args = get_args()

    # this binary parameter is put here first; we may want to set a scenario with no flexibility.
    fixed_load_bi = False

    lans_tlnd_df = pd.read_csv(f"{args.data_dir}/lans_inputs.csv")

    for index, row in lans_tlnd_df.iterrows():
        print(f"No {index} LAN-{row.UUID} starting running")
        ###  config  ###
        # 1: each node will built stand-alone generation system with no connection;
        # 2: all nodes are connected using TLND model, and then share one generation system
        config_list = [2]
        for config in config_list:
            # custom the output scenario name
            scenario_start_time = datetime.datetime.now()
            scenario_name = f"LAN_{row.UUID}_{row.Radius}m_config_{config}_XXs_limit"
            if os.path.exists(os.path.join(args.results_dir, scenario_name)):
                print(f"LAN {row.UUID} was already calculated, to rerun, delete the directory.")
                continue
            try:
                nodes_capacity_results = create_capacity_model(args, config, row)
                create_operation_model(args, nodes_capacity_results, scenario_name, config, row, scenario_start_time)
            except AttributeError:
                print(f"#### Error in LAN {row.UUID}")
    # showing the time used
    running_end_time = datetime.datetime.now()
    print(running_end_time - running_start_time)