import re
import os
##change = 1 -> a-b=-c to a minus b = "minus c"
## else-> a-b=-c to a + (-b) = -c


def replace_minus(content, change):
    parts = content.split("###")
    
    for i in range(len(parts)):
        question = parts[i]
        
        if change == 1:
            # Replace '-' with 'minus' in every question
            question = re.sub(r'(\d+)\s*-\s*(\d+)(?=\s*= answer\{)', r'\1 minus \2', question)
            question = re.sub(r'"answer":\s*-(\d+)}', r'"answer": "minus \1"}', question)
            question = re.sub(r'answer\{-(\d+)\}', r'answer{"minus \1"}', question)
        else:
            # Replace "-" with +(-b)
            question = re.sub(r'(\d+)\s*-\s*(\d+)(?=\s*= answer\{)', r'\1 + (-\2)', question)
        
        parts[i] = question
    
    return "###".join(parts)
    
    return content

def process_files(input_dir, output_dir, change):
    for root, dirs, files in os.walk(input_dir):
        if 'query_manipulation' in root:
            continue
        for file in files:
            if 'q_sub' in file:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                if change == 1:
                    output_subdir = os.path.join(output_dir, relative_path + "-minus")
                else:
                    output_subdir = os.path.join(output_dir, relative_path + "-bracket")
                os.makedirs(output_subdir, exist_ok=True)

                output_file_path = os.path.join(output_subdir, file)
                with open(file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
                    for line in infile:
                        modified_line = replace_minus(line.strip(), change)
                        outfile.write(modified_line + '\n')
                
                print(f"Processed and saved: {output_file_path}")

input_directory = r"understand_llm_math\exploration\std_op\data"
output_directory = r"understand_llm_math\exploration\std_op\data\query_manipulation"

process_files(input_directory, output_directory, change=1)

