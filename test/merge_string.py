def merge_strings(lst):
    result = []
    for string in sorted(lst, key=len, reverse=True):
        if not any(string in r for r in result):
            result.append(string)
    return result

lst = ['the person', 'the left of the image', 'the image', 'the left']
merged_lst = merge_strings(lst)
print(merged_lst)