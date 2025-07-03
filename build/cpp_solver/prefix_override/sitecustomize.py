import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rohan/new_ws/src/cpp_solver/install/cpp_solver'
