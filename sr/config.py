THRESHOLD_CONFIG = {
    'otsu_enabled': True,
    'adaptive_enabled': True,
    'manual_threshold': 127,
    'min_foreground_ratio': 0.1,
    'max_foreground_ratio': 0.8,
}


MORPHOLOGY_CONFIG = {
    'closing_enabled': True,
    'erosion_enabled': False,
    'dilation_enabled': False,
    'kernel_size': 3,
}


NOISE_REMOVAL_CONFIG = {
    'area_threshold_ratio': 0.2,
    'min_area_pixels': 50,
}


QUALITY_CONFIG = {
    'allowed_diff': 4,
    'defect_ratio_threshold_1': 0.01,
    'defect_ratio_threshold_2': 0.05,
    'max_defect_pixels': 100,
}


RADIUS_CONFIG = {
    'min_radius': 1,
    'radius_tolerance': 0.1,
}


DEBUG_CONFIG = {
    'show_intermediate_results': True,
    'save_intermediate_images': False,
    'verbose_output': True,
}

# MySQL 数据库配置（请根据实际环境修改）
MYSQL_CONFIG = {
    'host': '117.72.185.171',
    'port': 3306,
    'user': 'root',
    'password': 'YourPassword@123',
    'database': 'mxq',
    'charset': 'utf8mb4'
}