from capacity_model import create_capacity_model
from operation_model import create_operation_model
from fixed_load_model import create_fix_load_model
from utils import get_args
import datetime

if __name__ == '__main__':
    running_start_time = datetime.datetime.now()

    args = get_args()

    fixed_load_bi = False

    # 1 as stand alone, 2 as connected
    config_list = [1,2]
    for config in [2]:
        if fixed_load_bi:
            scenario_name = "Paloga-" + str(config) + '-fixed-load'
            create_fix_load_model(args, config)
        else:
            scenario_name = "Paloga-" + str(config) + '-flex-load'
            nodes_capacity_results = create_capacity_model(args, config)
            create_operation_model(args, nodes_capacity_results, scenario_name, config)


    running_end_time = datetime.datetime.now()
    print(running_end_time - running_start_time)