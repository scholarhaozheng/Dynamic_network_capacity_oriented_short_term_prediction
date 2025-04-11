import pickle
import numpy as np
import os

from metro_data_convertor.Find_project_root import Find_project_root


def Process_Time_DepartFreDic(station_index_root, Time_DepartFreDic_file_path, Time_DepartFreDic_Array_file_path):
    with open(Time_DepartFreDic_file_path, 'rb') as file:
        Time_DepartFreDic = pickle.load(file)

    with open(station_index_root, 'rb') as file:
        station_index = pickle.load(file)

    unique_stations = set()
    for station, index in station_index.items():
        unique_stations.add((station,index))

    sorted_stations = sorted(unique_stations, key=lambda x: x[1])
    station_indices = {name: idx for idx, (name, _) in enumerate(sorted_stations)}
    station_names = [station[0] for station in sorted_stations]
    sorted_lines = [1,2,3,4,5,44]
    line_indices = {line: idx for idx, line in enumerate(sorted_lines)}

    station_idx_get = station_indices.get
    line_idx_get = line_indices.get
    result_dict = {}
    """for timestamp, sections in Time_DepartFreDic.items():
        frequency_array = np.zeros((len(station_names), len(sorted_lines)))
        Index__+=1
        if not sections:
            result_dict[timestamp] = frequency_array
            continue
        for route, info in sections.items():
            start_station = info['start_station']['name']
            section_line = info['section_line']
            depart_freq = info['depart_freq']
            start_idx = station_idx_get(start_station)
            line_idx = line_idx_get(section_line)
            if depart_freq in {0, -1}:
                frequency_array[start_idx, line_idx] = 0
            else:
                freq_minutes = depart_freq.hour * 60 + depart_freq.minute
                frequency_array[start_idx, line_idx] = freq_minutes
        result_dict[timestamp] = frequency_array"""

    from concurrent.futures import ThreadPoolExecutor
    # 处理单个 timestamp 的函数，用于并行化
    def process_timestamp(item):
        timestamp, sections = item
        # 提前创建零矩阵
        frequency_array = np.zeros((len(station_names), len(sorted_lines)))

        # 如果 sections 为空，直接返回零矩阵
        if not sections:
            return timestamp, frequency_array

        # 解包并填充数组
        for route, info in sections.items():
            start_station = info['start_station']['name']
            section_line = info['section_line']
            depart_freq = info['depart_freq']

            # 查找索引
            start_idx = station_idx_get(start_station)
            line_idx = line_idx_get(section_line)

            # 计算发车频率
            if depart_freq in {0, -1}:
                frequency_array[start_idx, line_idx] = 0
            else:
                freq_minutes = depart_freq.hour * 60 + depart_freq.minute
                frequency_array[start_idx, line_idx] = freq_minutes
        # 返回结果
        return timestamp, frequency_array
    # 使用 ThreadPoolExecutor 并行处理
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_timestamp, Time_DepartFreDic.items()))
    result_dict = dict(results)

    with open(Time_DepartFreDic_Array_file_path, 'wb') as f:
        pickle.dump(result_dict, f)

"""project_root = Find_project_root()
Time_DepartFreDic_file_path = os.path.join(project_root, 'data', 'suzhou', 'Time_DepartFreDic.pkl')
Time_DepartFreDic_Array_file_path = os.path.join(project_root, 'data', 'suzhou', 'Time_DepartFreDic_Array.pkl')
Process_Time_DepartFreDic(Time_DepartFreDic_file_path, Time_DepartFreDic_Array_file_path)"""

