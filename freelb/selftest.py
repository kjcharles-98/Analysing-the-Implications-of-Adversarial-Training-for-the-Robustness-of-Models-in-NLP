import os

print('上级目录路径')
print(os.path.abspath(os.path.dirname(os.getcwd())))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(os.path.exists('/home'))
print(os.path.exists('/data/cheng/checkpoints'))