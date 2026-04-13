import re

def parse_model_structure(lines):
    """
    解析模型结构文本，构建树结构
    每行格式: 缩进 + (name): ClassName( 或 ) 或 空行
    """
    stack = []  # 存储 (缩进级别, 节点) 的栈
    root = {"name": "root", "children": [], "content": ""}
    stack.append((-1, root))
    
    for line_num, line in enumerate(lines):
        line = line.rstrip('\n')
        if not line.strip():
            continue
        
        # 计算缩进：开头的空格数
        indent = len(line) - len(line.lstrip())
        
        # 检查是否是关闭括号行
        if line.strip() == ')':
            # 弹出栈直到找到匹配的缩进级别
            while stack and stack[-1][0] >= indent:
                stack.pop()
            continue
        
        # 解析行内容: (name): ClassName( 或 其他
        match = re.match(r'^\s*\(([^)]+)\):\s*([^(]+)(?:\((.*)\))?', line)
        if not match:
            # 可能是其他格式，如 "24 x TransformerBlock("
            # 或者是没有括号的层，如 "SnakeBeta()"
            # 尝试匹配其他模式
            alt_match = re.match(r'^\s*([^:]+):\s*(.+)', line)
            if alt_match:
                name = alt_match.group(1).strip()
                class_part = alt_match.group(2).strip()
                node = {"name": name, "class": class_part, "children": [], "content": line}
            else:
                # 作为文本节点
                node = {"name": "", "class": "", "children": [], "content": line}
        else:
            name = match.group(1)
            class_name = match.group(2).strip()
            # 可能有额外的参数在括号中，如 "in_features=256, out_features=1536, bias=True"
            params = match.group(3) if match.group(3) else ""
            node = {"name": name, "class": class_name, "params": params, "children": [], "content": line}
        
        # 找到父节点：栈中缩进小于当前缩进的最后一个节点
        while stack and stack[-1][0] >= indent:
            stack.pop()
        
        parent_node = stack[-1][1]
        parent_node["children"].append(node)
        
        # 将当前节点压栈
        stack.append((indent, node))
    
    return root

def tree_to_markdown(node, level=0):
    """
    将树转换为Markdown列表
    """
    lines = []
    
    if level == 0:
        # 根节点，输出标题
        lines.append("# Stable Audio 模型结构\n")
        lines.append("> 基于 `stable-audio-tools` 库的模型架构\n")
        lines.append("\n## 模型概览\n")
        lines.append("以下是从 PyTorch 模型 `print()` 输出转换得到的结构化表示。\n")
        lines.append("\n## 详细结构\n")
        for child in node["children"]:
            lines.extend(tree_to_markdown(child, level + 1))
        return lines
    
    indent = "  " * (level - 1)
    
    # 如果有名称和类名
    if node.get("name") or node.get("class"):
        name_part = f"**{node['name']}**" if node.get("name") else ""
        class_part = f"`{node['class']}`" if node.get("class") else ""
        
        if name_part and class_part:
            line = f"{indent}- {name_part}: {class_part}"
        elif class_part:
            line = f"{indent}- {class_part}"
        else:
            line = f"{indent}- {node['content']}"
        
        # 添加参数
        if node.get("params"):
            line += f" ({node['params']})"
        
        lines.append(line)
    
    # 如果有子节点，递归处理
    for child in node["children"]:
        lines.extend(tree_to_markdown(child, level + 1))
    
    return lines

def convert_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 检查第一行是否有前导空格，如果有，去除所有行相同数量的前导空格
    if lines and lines[0].startswith(' '):
        # 计算最小缩进
        min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
        # 去除最小缩进
        lines = [line[min_indent:] for line in lines]
    
    tree = parse_model_structure(lines)
    
    markdown_lines = tree_to_markdown(tree)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_lines))
    
    print(f"转换完成，输出文件: {output_path}")

if __name__ == "__main__":
    input_file = "模型结构.md"
    output_file = "模型结构_易读.md"
    convert_file(input_file, output_file)