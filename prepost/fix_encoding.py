#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
编码修复工具
为所有Python文件添加正确的编码处理，确保在Windows环境下正常运行

作者: HydroSIS-2D Team
版本: 1.0
"""

import os
import sys
from pathlib import Path

def add_encoding_header(file_path):
    """为Python文件添加编码头"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 检查是否已有编码声明
    has_shebang = content.startswith('#!')
    has_encoding = '# -*- coding: utf-8 -*-' in content or '# coding: utf-8' in content
    
    if not has_encoding:
        lines = content.split('\n')
        insert_pos = 0
        
        # 如果有shebang，在其后插入
        if has_shebang:
            insert_pos = 1
        
        # 插入编码声明
        lines.insert(insert_pos, '# -*- coding: utf-8 -*-')
        content = '\n'.join(lines)
        
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        
        return True
    return False

def replace_unicode_chars(file_path):
    """替换特殊Unicode字符为ASCII等价物"""
    replacements = {
        '\u00b2': '^2',      # ²
        '\u00b3': '^3',      # ³
        '\u00b0': 'deg',     # °
        '\u00d7': 'x',       # ×
        '\u00f7': '/',       # ÷
        '\u00b1': '+/-',     # ±
        '\u2248': '~=',      # ≈
        '\u2264': '<=',      # ≤
        '\u2265': '>=',      # ≥
        '\u2260': '!=',      # ≠
        '\u2192': '->',      # →
        '\u2190': '<-',      # ←
        '\u2191': '^',       # ↑
        '\u2193': 'v',       # ↓
        '\u26a0': '[WARN]',  # ⚠
        '\u2713': '[OK]',    # ✓
        '\u2717': '[ERROR]', # ✗
    }
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    modified = False
    for unicode_char, ascii_equiv in replacements.items():
        if unicode_char in content:
            content = content.replace(unicode_char, ascii_equiv)
            modified = True
    
    if modified:
        with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
    
    return modified

def fix_encoding_in_directory(directory, extensions=('.py',)):
    """修复目录中所有Python文件的编码问题"""
    directory = Path(directory)
    fixed_files = []
    modified_files = []
    
    for file_path in directory.rglob('*'):
        if file_path.suffix in extensions and file_path.is_file():
            try:
                # 添加编码头
                if add_encoding_header(file_path):
                    fixed_files.append(str(file_path))
                
                # 替换Unicode字符
                if replace_unicode_chars(file_path):
                    modified_files.append(str(file_path))
                    
            except Exception as e:
                print(f"处理文件失败 {file_path}: {e}")
    
    return fixed_files, modified_files

def main():
    """主函数"""
    print("=" * 70)
    print("HydroSIS-2D 编码修复工具")
    print("=" * 70)
    
    # 获取prepost目录
    prepost_dir = Path(__file__).parent
    
    print(f"\n扫描目录: {prepost_dir}")
    print("正在修复编码问题...")
    
    # 修复编码
    fixed, modified = fix_encoding_in_directory(prepost_dir)
    
    print(f"\n完成!")
    print(f"  添加编码声明: {len(fixed)} 个文件")
    print(f"  替换Unicode字符: {len(modified)} 个文件")
    
    if modified:
        print("\n修改的文件:")
        for f in modified[:10]:  # 只显示前10个
            print(f"  - {Path(f).relative_to(prepost_dir)}")
        if len(modified) > 10:
            print(f"  ... 还有 {len(modified) - 10} 个文件")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()

