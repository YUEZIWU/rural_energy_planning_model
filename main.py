from capacity_model import create_capacity_model
from operation_model import create_operation_model
# from fixed_load_model import create_fix_load_model
from utils import get_args
import datetime

if __name__ == '__main__':
    running_start_time = datetime.datetime.now()

    args = get_args()

    # this binary parameter is put here first; we may want to set a scenario with no flexibility.
    fixed_load_bi = False

    ###  config  ###
    # 1: each node will built stand-alone generation system with no connection;
    # 2: all nodes are connected using TLND model, and then share one generation system
    config_list = [2]
    for config in config_list:
        # custom the output scenario name
        scenario_name = "Test-" + str(config) + '-20230419'
        nodes_capacity_results = create_capacity_model(args, config)
        create_operation_model(args, nodes_capacity_results, scenario_name, config)

    # showing the time used
    running_end_time = datetime.datetime.now()
    print(running_end_time - running_start_time)