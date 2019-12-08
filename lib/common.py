def count_lines(file):
    with open(file) as f:
        return sum(1 for line in f)
